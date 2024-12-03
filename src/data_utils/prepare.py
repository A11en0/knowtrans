from data_utils.data_loader import load_dataset

# load_dataset(dataset_name="abt_buy", task_type="EM", columns=[['name', 'description'],['name', 'description']], entity_name="product")

# load_dataset(dataset_name="phone", task_type="DI", columns=['Product Name'], entity_name="product", query_column="Brand Name")

# load_dataset(dataset_name="flights", task_type="ED", columns=['datasource','flight','scheduled departure time','actual departure time','scheduled arrival time','actual arrival time'])

load_dataset(dataset_name="cms", task_type="SM", columns=['name', 'description'])