# Imports
import featuretools as ft
import numpy as np
import pandas as pd

# Read data
app_train = pd.read_csv(r"C:\Users\HP\Documents\FDS\FDS Project\application_train.csv")
app_test = pd.read_csv(r"C:\Users\HP\Documents\FDS\FDS Project\application_test.csv")
bureau = pd.read_csv(r"C:\Users\HP\Documents\FDS\FDS Project\bureau.csv")
bureau_balance = pd.read_csv(r"C:\Users\HP\Documents\FDS\FDS Project\bureau_balance.csv")
pos = pd.read_csv(r"C:\Users\HP\Documents\FDS\FDS Project\POS_CASH_balance.csv")
credit = pd.read_csv(r"C:\Users\HP\Documents\FDS\FDS Project\credit_card_balance.csv")
previous = pd.read_csv(r"C:\Users\HP\Documents\FDS\FDS Project\previous_application.csv")
installment = pd.read_csv(r"C:\Users\HP\Documents\FDS\FDS Project\installments_payments.csv")

# We are now going to put app_train on top of app_test. This way we have all the data in one dataframe.
# Because we need to be able to understand from which dataframe comes each row we will make a new column
# that has as elements the original dataframe of that client. Also since app_test doesn't have the TARGET
# we will create it from the beginning and set it to null

app_train['set'] = 'train'
app_test['set'] = 'test'
app_test["TARGET"] = np.nan

# Now we will put train and then test. You can see that the first elements of the dataframe app is
# are the elements of app_train while the bottom ones are from app_test

app = app_train.append(app_test, ignore_index=True)

# What we want to do now is to gather all the information from the extra datasets that are given to us,
# so as to have as many features as possible to predict a more precise outcome. Even though we could
# perform this merge manually with the id columns ( SK_ID_CURR, SK_ID_BUREAU and SK_ID_PREV)
# there is a very useful tool to help us to automated feature engineering

# FEATURETOOLS

# Featuretools is a framework to perform automated feature engineering. It excels at transforming temporal
# and relational datasets into feature matrices for machine learning.
# Here we will use a method called DFS (Deep Feature Synthesis) that does automated feature engineering on relational
# (in our case) data

# We will now create our entity set.
# An EntitySet is a collection of entities and the relationships between them. They are useful for preparing raw,
# structured datasets for feature engineering. While many functions in Featuretools take entities and relationships as
# separate arguments, it is recommended to create an EntitySet, so you can more easily manipulate your data as needed.

# Creating an EntitySet
# First, we initialize an EntitySet. If you’d like to give it name, you can optionally provide an id to the constructor.

es = ft.EntitySet(id='clients')

# Adding entities
# To get started, we load the dataframes as an entity.

es = es.entity_from_dataframe(entity_id='app', dataframe=app, index='SK_ID_CURR')
es = es.entity_from_dataframe(entity_id='bureau', dataframe=bureau, index='SK_ID_BUREAU')
es = es.entity_from_dataframe(entity_id='previous', dataframe=previous, index='SK_ID_PREV')

# Notice how the id column has duplicates . If you try to create an entity with this Dataframe,
# you will run into an error.  make_index creates a unique index for each row by just looking at what number the row is,
# in relation to all the other rows. (it is like reset_index for pandas dataframe)

es = es.entity_from_dataframe(entity_id='bureau_balance', dataframe=bureau_balance,
                              make_index=True, index='bureaubalance_index')
es = es.entity_from_dataframe(entity_id='cash', dataframe=pos,
                              make_index=True, index='cash_index')
es = es.entity_from_dataframe(entity_id='installments', dataframe=installment,
                              make_index=True, index='installments_index')
es = es.entity_from_dataframe(entity_id='credit', dataframe=credit,
                              make_index=True, index='credit_index')
# Adding a Relationship
# We want to relate these entities by the id columns. Each product has multiple
# transactions associated with it, so it is called it the parent entity, while the transactions entity is known as the
# child entity. When specifying relationships we list the variable in the parent entity first. Note that each ft.
# Relationship must denote a one-to-many relationship rather than a relationship which is one-to-one or many-to-many.

# From the graph given to us we can see which are the relationships between our entities.

# relationship of bureau and bureau_balance via SK_ID_BUREAU
r_bureau_balance = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU'])

# relationship of bureau and app( application train and application test) via SK_ID_CURR
r_app_bureau = ft.Relationship(es['app']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])

# relationships from previous_applications  to POS_CASH_balance, installment_payments, credit_card_balance
# via SK_ID_PREV
r_previous_cash = ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'])
r_previous_installments = ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'])
r_previous_credit = ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])

# relationship between app(application_test, application_train) and previous_application via SK_ID_CURR

r_app_previous = ft.Relationship(es['app']['SK_ID_CURR'], es['previous']['SK_ID_CURR'])

# we add all these "links" together in our entity set

es = es.add_relationships([r_app_bureau, r_bureau_balance, r_app_previous,
                           r_previous_cash, r_previous_installments, r_previous_credit])

# Here we will use the DFG (Deep Feature Synthesis). We will give our entity set, the dataframe,where we want everything
# "sumed up" and how we want it "sumed up"(like a df.groupby(id).agg(["mean"])).
# We set as max_depth = 2 since we might need to aggregate 2 times (ex. from bureau_balance to bureau and from bureau
# to app)

feature_matrix, feature_names = ft.dfs(entityset=es, target_entity='app',
                                       agg_primitives=["mean"],
                                       max_depth=2, verbose=True)

# Since its a heavy computation we will save it in a csv for future use
feature_matrix.to_csv(r"C:\Users\HP\Documents\FDS\FDS Project\featurematrix.csv")
