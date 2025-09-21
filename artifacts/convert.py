usa_weight_file = "Mistral-7B-Instruct-v0.3.24K.20.500.pt"
weight_file = (
    "Mistral-7B-Instruct-v0.3.24K.20.500.hat_weights.pkl"
)
from sparse_attention_hub.sparse_attention.utils.hashattention_utils import create_hat_weights_file_from_usa
create_hat_weights_file_from_usa(usa_weight_file, weight_file, num_layers=32, num_heads=32, device="cpu")
