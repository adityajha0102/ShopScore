import logging
from typing import Tuple
from abc import ABC, abstractmethod
from collections import Counter
from typing import Union
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack


class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            # removing some mis filled data
            data = data[data['geolocation_state_y'] == data['seller_state']]

            # list of useless feature
            useless_features = ['review_comment_title', 'review_comment_message', 'product_category_name',
                                'product_weight_g', 'review_creation_date',
                                'product_length_cm', 'product_height_cm', 'product_width_cm', 'seller_city',
                                'review_answer_timestamp',
                                'geolocation_lat_y', 'geolocation_lng_y', 'geolocation_city_y', 'geolocation_state_y',
                                'review_id', 'order_approved_at', 'order_status',
                                'order_id', 'customer_id', 'order_item_id', 'geolocation_lat_x',
                                'geolocation_lng_x', 'geolocation_city_x', 'geolocation_state_x']
            data.drop(useless_features, axis=1, inplace=True)

            data.rename(columns={'product_category_name_english': 'product_category_name',
                                 'zip_code_prefix_x': 'zipCode_prefix_cust',
                                 'zip_code_prefix_y': 'zipCode_prefix_seller'}, inplace=True)

            data.dropna(how='any', inplace=True)

            return data
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e


class FeatureEngineeringStrategy(DataStrategy):
    """
    Strategy for feature engineer on the data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Creating and removing features
        """
        try:
            self.prepare_datetime(data)
            ix = data[(data['delivery_days'] > 60) | (data['estimated_days'] > 60) | (data['ships_in'] > 60)].index
            data.drop(ix, inplace=True)

            self.group_delivery(data)

            order_counts = [k for k, v in Counter(data.customer_unique_id).items() if v > 1]
            existing_cust = []
            for i in data.customer_unique_id.values:
                if i in order_counts:
                    existing_cust.append(1)
                else:
                    existing_cust.append(0)

            # seller popularity based on number of orders for each seller
            seller = data.seller_id.value_counts().to_dict()
            seller_popularity = []
            for _id in data.seller_id.values:
                seller_popularity.append(seller[_id])
            data['seller_popularity'] = seller_popularity
            data['existing_cust'] = existing_cust  # adding existing customer and seller_ID feature

            # if score> 3, set score = 1
            data.loc[data['review_score'] < 3, 'Score'] = 0
            data.loc[data['review_score'] > 3, 'Score'] = 1
            data.drop(data[data['review_score'] == 3].index, inplace=True)  # removing neutral reviews

            # removing all features we don't need
            data.drop(['customer_unique_id', 'seller_id', 'product_id', 'zipCode_prefix_seller', 'zipCode_prefix_cust',
                       'order_purchase_timestamp', 'order_delivered_carrier_date', 'review_score',
                                                                                   'order_delivered_customer_date',
                       'order_estimated_delivery_date', 'shipping_limit_date'], axis=1,
                      inplace=True)

            return data

        except Exception as e:
            logging.error("Error in feature engineering: {}".format(e))
            raise e

    def prepare_datetime(self, data: pd.DataFrame) -> pd.DataFrame:
        """
            Preparing the datetime data
        """
        try:
            # converting the timestamp format data to date data as we need just the date and not the exact time
            data['order_purchase_timestamp'] = pd.to_datetime(data['order_purchase_timestamp'], errors='coerce').dt.date
            data['order_estimated_delivery_date'] = pd.to_datetime(data['order_estimated_delivery_date'],
                                                                   errors='coerce').dt.date
            data['order_delivered_customer_date'] = pd.to_datetime(data['order_delivered_customer_date'],
                                                                   errors='coerce').dt.date
            data['shipping_limit_date'] = pd.to_datetime(data['shipping_limit_date'], errors='coerce').dt.date

            # subtracting the order_purchase_time to rest time based feature and converting date time into string to
            # remove the timestamp notation
            data['delivery_days'] = data['order_delivered_customer_date'].sub(data['order_purchase_timestamp'],
                                                                              axis=0).astype(str)
            data['estimated_days'] = data['order_estimated_delivery_date'].sub(data['order_purchase_timestamp'],
                                                                               axis=0).astype(str)
            data['ships_in'] = data['shipping_limit_date'].sub(data['order_purchase_timestamp'], axis=0).astype(str)

            data['delivery_days'] = data['delivery_days'].str.split(',').str.get(0)
            data['estimated_days'] = data['estimated_days'].str.split(',').str.get(0)
            data['ships_in'] = data['ships_in'].str.split(',').str.get(0)

            # converting type to int
            data['delivery_days'] = data['delivery_days'].str.extract(r'(\d+)').astype(float).astype(int)
            data['estimated_days'] = data['estimated_days'].str.replace(" days", "").astype(int)
            data['ships_in'] = data['ships_in'].str.replace(" days", "").astype(int)
            data['arrival_time'] = (data['estimated_days'] - data['delivery_days']).apply(
                lambda x: 'Early/OnTime' if x > 0 else 'Late')

            return data

        except Exception as e:
            logging.error("Error in feature engineering while preparing datetime data: {}".format(e))
            raise e

    def group_delivery(self, data: pd.DataFrame) -> pd.DataFrame:
        """
            grouping delivery time related columns
        """
        try:
            # binning and grouping delivery times into groups or classes

            delivery_feedbacks = []
            estimated_del_feedbacks = []
            shipping_feedback = []
            d_days = data.delivery_days.values.tolist()
            est_days = data.estimated_days.values.tolist()
            ship_days = data.ships_in.values.tolist()

            # actual delivery days
            for i in d_days:
                if i in range(0, 8):
                    delivery_feedbacks.append('Very_Fast')
                elif i in range(8, 16):
                    delivery_feedbacks.append('Fast')
                elif i in range(16, 25):
                    delivery_feedbacks.append('Neutral')
                elif i in range(25, 40):
                    delivery_feedbacks.append('Slow')
                elif i in range(40, 61):
                    delivery_feedbacks.append('Worst')

            # estimated delivery days
            for i in est_days:
                if i in range(0, 8):
                    estimated_del_feedbacks.append('Very_Fast')
                elif i in range(8, 16):
                    estimated_del_feedbacks.append('Fast')
                elif i in range(16, 25):
                    estimated_del_feedbacks.append('Neutral')
                elif i in range(25, 40):
                    estimated_del_feedbacks.append('Slow')
                elif i in range(40, 61):
                    estimated_del_feedbacks.append('Worst')

            # estimated shipping days
            for i in ship_days:
                if i in range(0, 4):
                    shipping_feedback.append('Very_Fast')
                elif i in range(4, 8):
                    shipping_feedback.append('Fast')
                elif i in range(8, 16):
                    shipping_feedback.append('Neutral')
                elif i in range(16, 28):
                    shipping_feedback.append('Slow')
                elif i in range(28, 61):
                    shipping_feedback.append('Worst')

            # putting list values into the dataframe as feature
            data['delivery_impression'] = delivery_feedbacks
            data['estimated_del_impression'] = estimated_del_feedbacks
            data['ship_impression'] = shipping_feedback

            return data

        except Exception as e:
            logging.error("Error in feature engineering while grouping delivery data: {}".format(e))
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Strategy to divide data into train and test
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            # splitting data to train and test data
            X = data.drop('Score', axis=1)
            Y = data.Score.values

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.15, stratify=Y, random_state=42)
            X_train_vec, X_test_vec = self.scale_data(X_train, X_test)

            return X_train_vec, X_test_vec, y_train, y_test

        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e

    def scale_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            std_scaler = Normalizer()
            min_max = MinMaxScaler()

            # payment_sequential feature
            payment_sequential_train = std_scaler.fit_transform(X_train.payment_sequential.values.reshape(-1, 1))
            payment_sequential_test = std_scaler.transform(X_test.payment_sequential.values.reshape(-1, 1))

            # payment_installments feature
            payment_installments_train = std_scaler.fit_transform(
                X_train.payment_installments.values.reshape(-1, 1))
            payment_installments_test = std_scaler.transform(X_test.payment_installments.values.reshape(-1, 1))

            # Payment value feature
            payment_value_train = std_scaler.fit_transform(X_train.payment_value.values.reshape(-1, 1))
            payment_value_test = std_scaler.transform(X_test.payment_value.values.reshape(-1, 1))

            # price
            price_train = std_scaler.fit_transform(X_train.price.values.reshape(-1, 1))
            price_test = std_scaler.transform(X_test.price.values.reshape(-1, 1))

            # freight_value
            freight_value_train = std_scaler.fit_transform(X_train.freight_value.values.reshape(-1, 1))
            freight_value_test = std_scaler.transform(X_test.freight_value.values.reshape(-1, 1))

            # product_name_length
            product_name_length_train = std_scaler.fit_transform(X_train.product_name_length.values.reshape(-1, 1))
            product_name_length_test = std_scaler.transform(X_test.product_name_length.values.reshape(-1, 1))

            # product_description_length
            product_description_length_train = std_scaler.fit_transform(
                X_train.product_description_length.values.reshape(-1, 1))
            product_description_length_test = std_scaler.transform(
                X_test.product_description_length.values.reshape(-1, 1))

            # product_photos_qty
            product_photos_qty_train = std_scaler.fit_transform(X_train.product_photos_qty.values.reshape(-1, 1))
            product_photos_qty_test = std_scaler.transform(X_test.product_photos_qty.values.reshape(-1, 1))

            # delivery_days
            delivery_days_train = std_scaler.fit_transform(X_train.delivery_days.values.reshape(-1, 1))
            delivery_days_test = std_scaler.transform(X_test.delivery_days.values.reshape(-1, 1))

            # estimated_days
            estimated_days_train = std_scaler.fit_transform(X_train.estimated_days.values.reshape(-1, 1))
            estimated_days_test = std_scaler.transform(X_test.estimated_days.values.reshape(-1, 1))

            # ships_in
            ships_in_train = std_scaler.fit_transform(X_train.ships_in.values.reshape(-1, 1))
            ships_in_test = std_scaler.transform(X_test.ships_in.values.reshape(-1, 1))

            # seller_popularity
            seller_popularity_train = min_max.fit_transform(X_train.seller_popularity.values.reshape(-1, 1))
            seller_popularity_test = min_max.transform(X_test.seller_popularity.values.reshape(-1, 1))

            # initialising oneHotEncoder

            onehot = CountVectorizer()
            cat = OneHotEncoder()
            # payment_type
            payment_type_train = onehot.fit_transform(X_train.payment_type.values)
            payment_type_test = onehot.transform(X_test.payment_type.values)

            # customer_state
            customer_state_train = onehot.fit_transform(X_train.customer_state.values)
            customer_state_test = onehot.transform(X_test.customer_state.values)

            # seller_state
            seller_state_train = onehot.fit_transform(X_train.seller_state.values)
            seller_state_test = onehot.transform(X_test.seller_state.values)

            # product_category_name
            product_category_name_train = onehot.fit_transform(X_train.product_category_name.values)
            product_category_name_test = onehot.transform(X_test.product_category_name.values)

            # arrival_time
            arrival_time_train = onehot.fit_transform(X_train.arrival_time.values)
            arrival_time_test = onehot.transform(X_test.arrival_time.values)

            # delivery_impression
            delivery_impression_train = onehot.fit_transform(X_train.delivery_impression.values)
            delivery_impression_test = onehot.transform(X_test.delivery_impression.values)

            # estimated_del_impression
            estimated_del_impression_train = onehot.fit_transform(X_train.estimated_del_impression.values)
            estimated_del_impression_test = onehot.transform(X_test.estimated_del_impression.values)

            # ship_impression
            ship_impression_train = onehot.fit_transform(X_train.ship_impression.values)
            ship_impression_test = onehot.transform(X_test.ship_impression.values)

            # existing_cust
            existing_cust_train = cat.fit_transform(X_train.existing_cust.values.reshape(-1, 1))
            existing_cust_test = cat.transform(X_test.existing_cust.values.reshape(-1, 1))

            # stacking up all the encoded features
            X_train_vec = hstack(
                (payment_sequential_train, payment_installments_train, payment_value_train, price_train,
                 freight_value_train, product_name_length_train, product_description_length_train,
                 product_photos_qty_train, delivery_days_train, estimated_days_train, ships_in_train,
                 payment_type_train, customer_state_train, seller_state_train, product_category_name_train,
                 arrival_time_train, delivery_impression_train, estimated_del_impression_train,
                 ship_impression_train, seller_popularity_train))

            X_test_vec = hstack((payment_sequential_test, payment_installments_test, payment_value_test, price_test,
                                 freight_value_test, product_name_length_test, product_description_length_test,
                                 product_photos_qty_test, delivery_days_test, estimated_days_test, ships_in_test,
                                 payment_type_test, customer_state_test, seller_state_test, product_category_name_test,
                                 arrival_time_test, delivery_impression_test, estimated_del_impression_test,
                                 ship_impression_test, seller_popularity_test))

            return X_train_vec, X_test_vec

        except Exception as e:
            logging.error("Error in dividing data in scaling step: {}".format(e))
            raise e


class DataCleaning:
    """
    Class for cleaning data which process the data and divides it into train and test
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle Data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e


if __name__ == "__main__":
    try:
        data = pd.read_csv(r"D:\Projects\ShopScore\data\merged_data\merged_data.csv")
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(data, process_strategy)
        processed_data = data_cleaning.handle_data()

        feature_strategy = FeatureEngineeringStrategy()
        data_cleaning = DataCleaning(processed_data, feature_strategy)
        feature_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(feature_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        print("Data Cleaning Completed")
        logging.info("Data Cleaning Completed")

    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e
