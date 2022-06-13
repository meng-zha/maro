import datetime
import pandas as pd

from maro.forecasting.MAROForecast.maro_forecasting import MAROForecasting
from .base_policy_data_loader import BaseDataLoader

class MAROForecastingDataLoader(BaseDataLoader):
    def __init__(self, data_loader_conf) -> None:
        super().__init__(data_loader_conf)

        self.data_loader_conf = data_loader_conf
        self.forecasting_conf = data_loader_conf["forecasting_conf"]
        self.forecaster = MAROForecasting(self.forecasting_conf)

        self.start_date = datetime.date.fromisoformat(data_loader_conf["start_date"])

        date_feature_path = self.data_loader_conf["date_feature_path"]
        self.date_feature = pd.read_csv(date_feature_path, parse_dates=["date"])
        self.items_info_path = self.data_loader_conf["items_info_path"]
        items_info = pd.read_csv(self.items_info_path)
        self.name2item_nbr = dict(zip(items_info["item_name"],items_info["item_id"]))

    def load(self, state: dict) -> pd.DataFrame:
        sku_name = state["sku_name"]
        item_nbr = self.name2item_nbr[sku_name]
        facility_name = state["facility_name"]
        store_nbr = int(facility_name.split("_")[1])
        hist_len = self.data_loader_conf["history_len"]
        fut_len = self.data_loader_conf["future_len"]

        history_offset = state["tick"] - self.data_loader_conf["history_len"]
        history_start_date = self.start_date + datetime.timedelta(history_offset)
        future_offset = state["tick"] + self.data_loader_conf["future_len"]
        future_end_date = self.start_date + datetime.timedelta(future_offset)

        # cut for history
        history_demand = state["history_demand"].tolist()
        history_price = state["history_price"].tolist()
        if len(history_demand) < self.data_loader_conf["history_len"]:
            history_demand = [pd.NA] * (hist_len - len(history_demand)) + history_demand
            history_price = [pd.NA] * (hist_len - len(history_price)) + history_price
        else:
            history_demand = history_demand[-self.data_loader_conf["history_len"]:]
            history_price = history_price[-self.data_loader_conf["history_len"]:]
        history_feature = pd.DataFrame({"date": pd.date_range(history_start_date, periods=self.data_loader_conf["history_len"]),
                                        "item_nbr": [item_nbr]*hist_len, "store_nbr": [store_nbr]*fut_len,
                                        'total_qty': history_demand, 'avg_price': history_price})
        
        # cut for future feature
        future_feature = self.date_feature[(self.date_feature["date"] >= pd.to_datetime(history_start_date)) \
                                        & (self.date_feature["date"] <= pd.to_datetime(future_end_date))]
        input_tensor = pd.merge(history_feature, future_feature, on='date', how='right')
        input_tensor['item_nbr'].ffill(inplace=True)
        input_tensor['store_nbr'].ffill(inplace=True)
        predict_df = self.forecaster.predict({'rectified_sku60':input_tensor})
        predict_df = predict_df[(predict_df['item_nbr'] == item_nbr)\
            & (predict_df['store_nbr'] == store_nbr)]
        
        target_df = pd.DataFrame(columns=["Price", "Cost", "Demand"])
        # Including history and today
        history_start = max(state["tick"] - self.data_loader_conf["history_len"], 0)
        for index in range(history_start, state["tick"] + 1):
            target_df = target_df.append(pd.Series({
                "Price": state["history_price"][index],
                "Cost": state["unit_order_cost"],
                "Demand": state["history_demand"][index]
            }), ignore_index=True)
        his_mean_price = target_df["Price"].mean().item()
        for index in range(0, fut_len):
            target_df = target_df.append(pd.Series({
                "Price": his_mean_price,
                "Cost": state["unit_order_cost"],
                "Demand": predict_df[f"{index}"].item(),
            }), ignore_index=True)
        return target_df