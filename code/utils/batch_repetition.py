# author: Nikola Zubic

import torch


def repeat_tensor_for_each_element_in_batch(torch_tensor, n):
    """
    Repeats a certain torch tensor n times for each element in a batch.

    :param torch_tensor: given torch tensor
    :param n: number of repeats
    :return: new tensor, where every row of torch_tensor is repeated n times
    """
    data_shape = torch_tensor.shape[1:]  # 3
    repeats = [1, n] + [1] * len(data_shape)

    expanded = torch_tensor.unsqueeze(1).repeat(*repeats)

    return expanded.view(-1, *data_shape)


if __name__ == "__main__":
    tensor = torch.rand(2, 3)
    print("Before repeat:\n" + str(tensor))

    repetition = repeat_tensor_for_each_element_in_batch(torch_tensor=tensor, n=3)

    print("After repeat:\n" + str(repetition))
