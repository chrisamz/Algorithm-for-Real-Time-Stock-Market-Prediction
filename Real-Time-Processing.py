# real_time_processing.py

"""
Real-Time Processing Module for Real-Time Stock Market Prediction

This module contains functions for real-time data ingestion, processing, and integration
to adapt to changing market conditions and make immediate stock market predictions.

Tools Used:
- Apache Kafka
- Apache Spark
- Real-time databases
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_json
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
import kafka
from kafka import KafkaProducer, KafkaConsumer
import json

class RealTimeProcessing:
    def __init__(self, kafka_bootstrap_servers='localhost:9092', topic='stock_market_data'):
        """
        Initialize the RealTimeProcessing class.
        
        :param kafka_bootstrap_servers: str, Kafka bootstrap servers
        :param topic: str, Kafka topic to consume and produce data
        """
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.topic = topic
        self.spark = SparkSession.builder.appName("RealTimeStockMarketPrediction").getOrCreate()
        
        self.schema = StructType([
            StructField("timestamp", TimestampType(), True),
            StructField("open", DoubleType(), True),
            StructField("high", DoubleType(), True),
            StructField("low", DoubleType(), True),
            StructField("close", DoubleType(), True),
            StructField("volume", DoubleType(), True),
            StructField("sentiment", DoubleType(), True)
        ])

    def read_from_kafka(self):
        """
        Read real-time data from Kafka topic.
        
        :return: DataFrame, real-time data
        """
        return self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers) \
            .option("subscribe", self.topic) \
            .load() \
            .selectExpr("CAST(value AS STRING)") \
            .select(
                from_json(col("value"), self.schema).alias("data")
            ) \
            .select("data.*")

    def write_to_kafka(self, df):
        """
        Write processed data to Kafka topic.
        
        :param df: DataFrame, processed data
        """
        df.selectExpr("to_json(struct(*)) AS value") \
            .writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers) \
            .option("topic", self.topic) \
            .option("checkpointLocation", "/tmp/kafka_checkpoint") \
            .start() \
            .awaitTermination()

    def process_data(self, df):
        """
        Process real-time data to extract relevant features and perform necessary transformations.
        
        :param df: DataFrame, input data
        :return: DataFrame, processed data
        """
        df = df.withColumn("hour", col("timestamp").cast("timestamp").cast("long") // 3600)
        return df

    def start_processing(self):
        """
        Start the real-time data processing pipeline.
        """
        raw_data = self.read_from_kafka()
        processed_data = self.process_data(raw_data)
        self.write_to_kafka(processed_data)

if __name__ == "__main__":
    rtp = RealTimeProcessing()
    rtp.start_processing()
