import torch
from mojo_opset.utils.hf_utils import (
    create_renaming_by_dict,
    load_weights_with_renaming_and_converter,
)


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_list = torch.nn.ModuleList([torch.nn.Linear(1024, 1024)] * 3)
        self.param1 = torch.nn.Parameter(torch.empty(1024))
        self.param2 = torch.nn.Parameter(torch.empty(1024))


class FlattenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(1024, 1024)
        self.fc1 = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.concat_param = torch.nn.Parameter(torch.empty(2048))


def test_weight_loading():
    # odict_keys(['param1', 'param2', 'fc_list.0.weight', 'fc_list.0.bias', 'fc_list.1.weight', 'fc_list.1.bias', 'fc_list.2.weight', 'fc_list.2.bias'])
    model_a = SimpleModel()
    # odict_keys(['concat_param', 'fc0.weight', 'fc0.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])
    model_b = FlattenModel()

    for p in model_a.parameters():
        torch.nn.init.ones_(p)
    for p in model_b.parameters():
        torch.nn.init.zeros_(p)

    name_mapping = {"fc_list\.(.*)\.": "fc(\1)."}
    weight_renaming = create_renaming_by_dict(name_mapping)

    from transformers.core_model_loading import Concatenate, WeightConverter

    weight_converter = WeightConverter(["param1", "param2"], ["concat_param"], operations=[Concatenate(dim=0)])

    load_weights_with_renaming_and_converter(
        model_b, model_a.state_dict(), renamings=weight_renaming, converters=[weight_converter]
    )

    for p in model_b.parameters():
        expected = torch.ones_like(p)
        assert torch.all(p == expected)
