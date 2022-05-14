from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import json
import pandas as pd

df=pd.read_csv('C:\\Users\\1672040\Desktop\project\Phishing project\data\Validated_train_data\\train.csv')


cloud_config= {
        'secure_connect_bundle': 'C:\\Users\\1672040\\Downloads\\secure-connect-ashish.zip'
}
auth_provider = PlainTextAuthProvider('cZhGNGzSrZmCtMnguMbZcBvd', '0HbS-wCn.gz8v33F4AUyWCNbZv_OKnru.6Bi6bocR8wXTi99gr3z1TWH8gG9YhrD2aOu5qMDnZ5F3IE2IXSNwNj_9ZHD30_Mv3K9IfJEyE_fhUAEqt,EtuXs6Ps0eD3B')
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()
session.execute('DESCRIBE keyspaces;')
#session.execute("CREATE KEYSPACE IF NOT EXISTS phising1 WITH replication = {'class':'SimpleStrategy', 'replication_factor' : 3};")

#row = session.execute(f"COPY Data {tuple(df.columns)} FROM 'C:\\Users\\1672040\Desktop\project\Phishing project\data\Validated_train_data\\train.csv' WITH HEADER = TRUE; ")

