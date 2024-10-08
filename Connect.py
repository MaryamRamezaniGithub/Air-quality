#!/usr/bin/env python
# coding: utf-8

# <font size="+3"><strong> Extract Data from MongoDB </strong></font>

from pprint import PrettyPrinter
import pandas as pd
from pymongo import MongoClient

pp =PrettyPrinter(indent=2)


# ## Connect

client = MongoClient(host="localhost", port=27017)

pp.pprint(list(client.list_databases()))

list(client.list_databases())[1]["name"]

db = client["air-quality"]

for c in list(db.list_collections()):
    print(c["name"])

nairobi = db["nairobi"]

nairobi.count_documents({})

result =nairobi.find_one({})
pp.pprint(result)


# How many sites are in a document?



nairobi.distinct("metadata.site")

# How many documents are associated with each site?

print("Documents from site 6:", nairobi.count_documents({"metadata.site":6}))
print("Documents from site 29:", nairobi.count_documents({"metadata.site":29}))


#   How many readings there are for each site in the nairobi collection?
result =nairobi.aggregate([
    { "$group": {"_id" : "$metadata.site" , "count":{"$count":{}}
                }
    }
    
])
pp.pprint(list(result))

# How many types of measurements have been taken in the nairobi collection
nairobi.distinct("metadata.measurement")

# Retrieve the $PM 2.5$ readings from all sites

result =nairobi.find({"metadata.measurement": "P2"}).limit(3)
pp.pprint(list(result))

# How many readings there are for each type ("humidity", "temperature", "P2", and "P1") in site 29

result =nairobi.aggregate([
    {"$match":{"metadata.site":29}},
    {"$group":{"_id":"$metadata.measurement", "count":{"$count":{}}}}
    
])
pp.pprint(list(result))


# ## Import into a Pandas DataFrame



result = nairobi.find({"metadata.site":29, "metadata.measurement":"P2"},
                      projection={"P2":1, "timestamp":1, "_id":0}
)


pp.pprint(next(result))

type(next(result))

df1=pd.DataFrame.from_dict(result).set_index("timestamp")
df1.head()

# Check our work
assert df.shape[1] == 1, f"`df` should have only one column, not {df.shape[1]}."
assert df.columns == [
    "P2"
], f"The single column in `df` should be `'P2'`, not {df.columns[0]}."
assert isinstance(df.index, pd.DatetimeIndex), "`df` should have a `DatetimeIndex`."





