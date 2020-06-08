from model import contrastive_volume_loss
import torch
from torch.nn.functional import normalize
def one_point_same():
    emb_0 = normalize(torch.tensor([[[[[-1,1],[1,1]],[[1,1],[1,1]]]]], dtype=torch.float32), dim=1)
    emb_1 = normalize(torch.tensor([[[[[-1,1], [1,1]],[[1,1],[1,1]]]]], dtype=torch.float32), dim=1)
    
    locations_0 = torch.tensor([[[0, 0, 0]]], dtype=torch.float32)
    locations_1 = torch.tensor([[[0, 0, 0]]], dtype=torch.float32)
    
    print(emb_0.size())
    print(emb_1.size())
    print(locations_0.size())
    print(locations_1.size())

    actual = contrastive_volume_loss(emb_0, emb_1, locations_0, locations_1, 1).numpy()
    expected = 0.0

    print("Pass: " + str(actual == expected))
    print("     Actual: " + str(actual) + " Expected: " + str(expected))

def two_point_diff():
    emb_0 = normalize(torch.tensor([[[[1,-1],[1,1]]]], dtype=torch.float32), dim=1)
    emb_1 = normalize(torch.tensor([[[[-1,1],[1,1]]]], dtype=torch.float32), dim=1)

    locations_0 = torch.tensor([[[0, 0], [0,1]]], dtype=torch.float32)
    locations_1 = torch.tensor([[[0, 0], [0,1]]], dtype=torch.float32)

    print(emb_0.size())
    print(emb_1.size())
    print(locations_0.size())
    print(locations_1.size())

    actual = contrastive_volume_loss(emb_0, emb_1, locations_0, locations_1, 1).numpy()
    expected = 2.2395448

    print("Pass: " + str(actual == expected))
    print("     Actual: " + str(actual) + " Expected: " + str(expected))

def one_point_diff():
    emb_0 = normalize(torch.tensor([[[[1,-1],[1,1]]]], dtype=torch.float32), dim=1)
    emb_1 = normalize(torch.tensor([[[[-1,1],[1,1]]]], dtype=torch.float32), dim=1)

    locations_0 = torch.tensor([[[0, 0]]], dtype=torch.float32)
    locations_1 = torch.tensor([[[0, 0]]], dtype=torch.float32)

    print(emb_0.size())
    print(emb_1.size())
    print(locations_0.size())
    print(locations_1.size())

    actual = contrastive_volume_loss(emb_0, emb_1, locations_0, locations_1, 1).numpy()
    expected = 0

    print("Pass: " + str(actual == expected))
    print("     Actual: " + str(actual) + " Expected: " + str(expected))

def multi_channel_diff():
    emb_0 = normalize(torch.tensor([[[[1,-1],[1,1]], [[1,-1],[1,1]]]], dtype=torch.float32), dim=1)
    emb_1 = normalize(torch.tensor([[[[-1,1],[1,1]], [[1,-1],[1,1]]]], dtype=torch.float32), dim=1)

    locations_0 = torch.tensor([[[0, 0], [0,1]]], dtype=torch.float32)
    locations_1 = torch.tensor([[[0, 0], [0,1]]], dtype=torch.float32)
    
    print(emb_0)
    print(emb_1)
    print(emb_0.size())
    print(emb_1.size())
    print(locations_0.size())
    print(locations_1.size())

    actual = contrastive_volume_loss(emb_0, emb_1, locations_0, locations_1, 1).numpy()
    expected = 2.2395448

    print("Pass: " + str(actual == expected))
    print("     Actual: " + str(actual) + " Expected: " + str(expected))


one_point_same()
one_point_diff()
two_point_diff()
multi_channel_diff()
