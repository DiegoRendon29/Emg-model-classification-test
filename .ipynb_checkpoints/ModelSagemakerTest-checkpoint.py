from ETL import *
import boto3

import json
import math
from emgObject import Movement
import scipy.io
import io
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
import os
import pandas as pd
import numpy as np
from botocore.exceptions import ClientError
import os
from pyspark.conf import SparkConf
pd.DataFrame.iteritems = pd.DataFrame.items

def separte_data():
    pass

if __name__ == "__main__":
    df = obtain_complete_data(size_frame=31)