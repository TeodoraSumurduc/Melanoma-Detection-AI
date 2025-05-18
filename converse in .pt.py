import torch
from nn_class import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Creezi modelul
model = Net().to(device)

# 2. Încarci state_dict-ul
state_dict = torch.load("melanoma_model.pth", map_location=device)
model.load_state_dict(state_dict)

# 3. Treci în modul evaluare
model.eval()

# 4. Salvezi modelul TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")  # sau .ptl dacă preferi
