import os
from dataset.data_utility.dataset_utility import load_from_file
from fore_test.evaluator.base_evaluator import BaseEvaluator
from fore_test.test_utility.test_data_utility import *

class WalmartEvaluator(BaseEvaluator):
    def __init__(self, conf, test_loader, model):
        super().__init__(conf, test_loader, model)

    def build_predict(self):
        self.predict_pf = super().build_predict()

        self.predict_pf['weekly_predict'] = self.predict_pf['predict'].copy(deep=True)
        self.predict_pf['date'] = self.predict_pf['date'] + datetime.timedelta(1) 

    def build_ground_truth(self):
        date_column = self.conf.data.shared_columns.date
        target_column = self.conf.data.shared_columns.target
        node_columns = self.conf.data.shared_columns.nodes
        start_date = self.test_loader.dataset.start_date
        end_date = self.test_loader.dataset.end_date
        
        ground_files_path = [os.path.join(self.conf.data.root_path, file) for file in self.conf.test.ground_truth.files]
        df_total = load_from_file(ground_files_path, parse_date=date_column).sort_values(by=[date_column])
        min_date = df_total[date_column].min()
        max_date = df_total[date_column].max()
    
        index = node_columns + [date_column]
        df_total = df_total.set_index(index)
        df_target = df_total[target_column].unstack()

        his_dates = self.conf.data.sequence_feature.history.dates
        fut_dates = self.conf.data.sequence_feature.future.dates
        abandon_dates = self.conf.test.ground_truth.abandon_dates
        history_mean_dates = self.conf.test.ground_truth.history_mean_dates
        df_target = fill_zero_target_with_history_mean(df_target, pd.to_datetime(start_date), pd.to_datetime(end_date), abandon_dates, history_mean_dates)
        df_l4week = get_history_mean(df_target.copy(), min_date, max_date, his_dates)
        ground_truth_pf = pd.concat([df_target.stack(),df_l4week.stack()],axis=1,keys=['ground_truth','last_4_week_mean']).reset_index()

        ground_truth_pf = ground_truth_pf[ \
            (ground_truth_pf['date'] >= pd.to_datetime(start_date) + datetime.timedelta(his_dates)) &\
            (ground_truth_pf['date'] <= pd.to_datetime(end_date) - datetime.timedelta(fut_dates-1))]
        self.ground_truth_pf = ground_truth_pf
    
    def eval(self):
        super().eval()
        eval_conf = self.conf.test.eval
        abandon_store = eval_conf.abandon_store
        
        store_list = []
        main_store_conf = eval_conf.main_store
        main_store_dic = {item.main: item.sub for item in main_store_conf}

        for main_store in main_store_dic.keys():
            store_list += main_store_dic[main_store]
        store_list = [int(s) for s in store_list if int(s) not in abandon_store]
        store_list.sort()

        # store daily
        date_column = self.conf.data.shared_columns.date
        node_columns = self.conf.data.shared_columns.nodes
        index = node_columns + [date_column]
        df_merge = pd.merge(self.ground_truth_pf.set_index(index), self.predict_pf.set_index(index),\
            how='outer', left_index=True, right_index=True).reset_index()
        df_merge = df_merge[~df_merge['store_nbr'].isin(abandon_store)]
        

        stores_acc = {}
        # main results
        df_merge_main = df_merge.copy()
        for main_store, related_list in main_store_dic.items():
            df_merge_main.loc[df_merge_main['store_nbr'].isin(related_list),'store_nbr'] = main_store
        
        df_merge_main = df_merge_main.groupby(by=index).sum().reset_index()
        for main_store, related_list in main_store_dic.items():
            tmp = df_merge_main[df_merge_main['store_nbr'] == main_store]
            tmp_weekly_sum = tmp.groupby('item_nbr').resample('W-Fri',on='date').sum()
            tmp_weekly_last_4_week_mean = tmp.groupby('item_nbr').resample('W-Fri',on='date').first().fillna(0)
            daily_acc = self.cal_main_store_acc(tmp['ground_truth'], tmp['predict'], tmp['last_4_week_mean'])
            weekly_acc = self.cal_main_store_acc(tmp_weekly_sum['ground_truth'], tmp_weekly_sum['weekly_predict'], tmp_weekly_last_4_week_mean['last_4_week_mean'])
            stores_acc[main_store] = {'daily': daily_acc, 'weekly': weekly_acc}

        for inv_store in store_list:
            tmp = df_merge[df_merge['store_nbr'] == inv_store]
            tmp = tmp[~tmp['predict'].isna()]
            tmp = tmp[~tmp['ground_truth'].isna()]
            tmp_weekly_sum = tmp.groupby('item_nbr').resample('W-Fri',on='date').sum()
            daily_acc = self.cal_inv_store_acc(tmp['ground_truth'], tmp['predict'])
            weekly_acc = self.cal_inv_store_acc(tmp_weekly_sum['ground_truth'], tmp_weekly_sum['weekly_predict'])
            if inv_store not in main_store_dic.keys():
                stores_acc[inv_store] = {'daily': daily_acc, 'weekly': weekly_acc}

        # item results
        item_acc = {}
        df_merge = df_merge.groupby(by=['item_nbr','date']).sum().reset_index()
        for item, item_df in df_merge.groupby('item_nbr'):
            tmp_weekly_sum = item_df.resample('W-Fri',on='date').sum()
            daily_acc = self.cal_inv_store_acc(item_df['ground_truth'], item_df['predict'])
            weekly_acc = self.cal_inv_store_acc(tmp_weekly_sum['ground_truth'], tmp_weekly_sum['weekly_predict'])
            item_acc[item] =  {'daily': daily_acc, 'weekly': weekly_acc}

        result_path = os.path.join(self.conf.log_dir, self.conf.experiment.experiment_name + '_' + eval_conf.predict_result)
        if os.path.exists(result_path):
            pd.concat([pd.read_csv(result_path, parse_dates=[date_column]), self.predict_pf]).to_csv(result_path,index=False)
        else:
            self.predict_pf.to_csv(result_path,index=False)
        pd.DataFrame(stores_acc).to_csv(os.path.join(self.conf.log_dir, eval_conf.accuracy_result))
        return stores_acc
    
    def cal_main_store_acc(self, pred, actual, l4week):
        esp = 1e-9
        pred = np.array(pred)
        actual = np.array(actual)
        l4week = np.array(l4week)
        smape = 1 - 2 * (np.abs(pred-actual)) / (pred+actual + esp)
        return (smape * l4week).sum() / (l4week.sum() + esp)

    def cal_inv_store_acc(self, pred, actual):
        esp = 1e-9
        pred = np.array(pred)
        actual = np.array(actual)
        smape = 2 * (np.abs(pred - actual).sum()) / ((pred+actual).sum() + esp)
        return 1 - smape