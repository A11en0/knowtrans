import random


# def subsample_data(data, subsample_size):
#     """
#     Subsample data. Data is in the form of a tuple of lists.
#     """
#     inputs, outputs = data
#     assert len(inputs) == len(outputs)
#     data_size = len(inputs)
#     if subsample_size == -1:
#         subsample_size = data_size
#     elif subsample_size > data_size:
#         subsample_size = data_size
        
#     indices = random.sample(range(data_size), subsample_size)
#     inputs = [inputs[i] for i in indices]
#     outputs = [outputs[i] for i in indices]
#     return inputs, outputs
def subsample_data(data, subsample_size):
    """
    Subsample data. Data is in the form of a tuple of lists.
    """
    data_size = len(data)
    if subsample_size == -1 or subsample_size >= data_size:
        return data
    else:
        indices = random.sample(range(data_size), subsample_size)
        return [data[i] for i in indices]
    


# def create_split(data, split_size):
#     """
#     Split data into two parts. Data is in the form of a tuple of lists.
#     """
#     inputs, outputs = data
#     assert len(inputs) == len(outputs)

#     random.seed(42)
#     indices = random.sample(range(len(inputs)), split_size)
#     inputs1 = [inputs[i] for i in indices]
#     outputs1 = [outputs[i] for i in indices]
#     inputs2 = [inputs[i] for i in range(len(inputs)) if i not in indices]
#     outputs2 = [outputs[i] for i in range(len(inputs)) if i not in indices]
#     return (inputs1, outputs1), (inputs2, outputs2)

def create_split(data, split_size):
    """
    Split data into two parts. Data is in the form of a tuple of lists.
    """
    random.seed(42)
    indices = random.sample(range(len(data)), split_size)
    data1 = [data[i] for i in indices]
    data2 = [data[i] for i in range(len(data)) if i not in indices]
    return data1, data2
