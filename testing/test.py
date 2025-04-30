import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import os

CARRIER_FREQ = 28e9
LIGHT_SPEED = 3e8
WAVELENGTH = LIGHT_SPEED / CARRIER_FREQ
D = WAVELENGTH / 2
NR1, NR2 = 32, 32
NR = NR1 * NR2
NRF = 16

def get_Wr(device=None):
    F1 = torch.fft.fft(torch.eye(NR1, device=device))
    F2 = torch.fft.fft(torch.eye(NR2, device=device))
    return torch.kron(F1, F2) / np.sqrt(NR)

def generate_positions():
    uR = np.random.uniform(-1000, 1000, size=(3,))
    uU = np.random.uniform(-1000, 1000, size=(3,))
    return uR, uU

def generate_channel(uR, uU):
    H = np.zeros((NR1, NR2), dtype=complex)
    for i in range(NR1):
        for j in range(NR2):
            disp = D * np.array([i, 0, j])
            dist = np.linalg.norm(uR + disp - uU)
            H[i,j] = (WAVELENGTH/(4*np.pi*dist)) * np.exp(-2j*np.pi*dist/WAVELENGTH)
    return H

class UAVChannelDataset(Dataset):
    def __init__(self, num_samples, snr_db=0, T=32):
        self.num_samples = num_samples
        self.snr_db = snr_db
        self.T = T
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        uR, uU = generate_positions()
        H = generate_channel(uR, uU)
        Wr = get_Wr()
        H_vec = torch.from_numpy(H.reshape(-1,1)).to(torch.complex64)
        H_sparse = Wr.conj().T @ H_vec

        pH = torch.norm(H_sparse)
        H_sparse = H_sparse / pH

        A_list, y_list = [], []
        for _ in range(self.T):
            A = (2*torch.randint(0,2,(NRF,NR),dtype=torch.float32) - 1).to(torch.complex64)
            A = A / np.sqrt(NRF * self.T)  
            y_clean = A @ (Wr @ H_sparse)
            noise_power = torch.mean(torch.abs(y_clean)**2) / (10**(self.snr_db/10))
            noise = (torch.randn_like(y_clean) + 1j*torch.randn_like(y_clean)) * np.sqrt(noise_power/2)
            y = y_clean + noise
            A_list.append(A)
            y_list.append(y)
        A_bar = torch.cat(A_list,dim=0)
        y_bar = torch.cat(y_list,dim=0)

        y_ri = torch.view_as_real(y_bar)
        A_ri = torch.view_as_real(A_bar)
        H_ri = torch.view_as_real(H_sparse)
        return y_ri, A_ri, H_ri

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(x)))

def soft_threshold_complex(x, lam):
    nr = x.size(-1)//2
    real, imag = x[..., :nr], x[..., nr:]
    mag = torch.sqrt(real**2 + imag**2 + 1e-6)
    scale = torch.clamp(mag - lam, min=0)/mag
    return torch.cat([real*scale, imag*scale], dim=-1)

class RADMMLayer(nn.Module):
    def __init__(self, nr):
        super().__init__()
        dim = 2*nr
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta  = nn.Parameter(torch.tensor(0.1))
        self.gamma1= nn.Parameter(torch.tensor(0.1))
        self.gamma2= nn.Parameter(torch.tensor(0.1))
        self.eta1  = nn.Parameter(torch.tensor(0.1))
        self.eta2  = nn.Parameter(torch.tensor(0.1))
        self.rho1  = nn.Parameter(torch.tensor(0.1))
        self.rho2  = nn.Parameter(torch.tensor(0.1))
        self.lam   = nn.Parameter(torch.tensor(0.1))
        self.res1  = ResidualBlock(dim)
        self.res2  = ResidualBlock(dim)
        self.nr    = nr

    def forward(self, H, X, u, A, Y, Wr):
        B = A.size(0)
        ATA = A.transpose(1,2) @ A
        ATY = A.transpose(1,2) @ Y
        I = torch.eye(self.nr, device=ATA.device).unsqueeze(0).repeat(B,1,1)
        Hn = torch.linalg.inv(ATA + self.beta*I) @ (ATY + self.alpha*(X-u))
        def ri(z): return torch.cat([z.real, z.imag], dim=1)
        def crim(v): return v[..., :self.nr] + 1j*v[..., self.nr:]
        d1 = X*(1-2*self.eta1*self.rho1) + Hn*(2*self.eta1*self.rho1)
        z1 = self.res1(ri(d1.squeeze(-1))).view(B, -1)
        t1 = soft_threshold_complex(z1, self.lam)
        t1c = crim(t1)
        d2 = t1c*(1-2*self.eta2*self.rho2) + Hn.squeeze(-1)*(2*self.eta2*self.rho2)
        z2 = self.res2(ri(d2).view(B, -1))
        Xn = crim(z2).unsqueeze(-1)
        un = u + self.gamma1*Hn - self.gamma2*Xn
        return Hn, Xn, un

class RADMMNet(nn.Module):
    def __init__(self, num_layers, nr):
        super().__init__()
        self.layers = nn.ModuleList([RADMMLayer(nr) for _ in range(num_layers)])
    def forward(self, Y, A, Wr):
        B = Y.size(0)
        H = torch.zeros((B, NR, 1), dtype=torch.cfloat, device=Y.device)
        X = torch.zeros_like(H)
        u = torch.zeros_like(H)
        for layer in self.layers:
            H, X, u = layer(H, X, u, A, Y, Wr)
        return Wr @ H

def NMSE(true, pred):
    return torch.mean(torch.norm(true - pred, dim=(-2,-1))**2 / torch.norm(true, dim=(-2,-1))**2)

def train_one_model(T, SNR_dB, num_layers):
    print(f"\nTraining: T={T}, SNR={SNR_dB}dB, Layers={num_layers}")

    dataset = UAVChannelDataset(num_samples=1000, snr_db=SNR_dB, T=T)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    model   = RADMMNet(num_layers=num_layers, nr=NR).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    Wr = get_Wr(device)
    loss_history = []

    for epoch in range(3):
        total_loss = 0
        for y_ri, A_ri, H_ri in loader:
            y  = torch.view_as_complex(y_ri).to(device)
            A  = torch.view_as_complex(A_ri).to(device)
            Hs = torch.view_as_complex(H_ri).to(device)
            pred = model(y, A, Wr)
            loss = NMSE(Hs, pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        nmse_db = 10 * np.log10(avg_loss/T + 1e-12)
        print(f"Epoch {epoch+1}/50: NMSE = {nmse_db:.2f} dB")

    os.makedirs('checkpoints', exist_ok=True)
    save_path = f'../New_Model/checkpoints/RADMM_layers{num_layers}_nr1{NR1}_NR2{NR2}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return loss_history



def test_model(checkpoint_path, T, SNR_dB, num_layers, num_samples=200):
    print(f"Testing: {checkpoint_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Wr = get_Wr(device)

    dataset = UAVChannelDataset(num_samples=num_samples, snr_db=SNR_dB, T=T)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = RADMMNet(num_layers=num_layers, nr=NR).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    total_nmse = 0
    with torch.no_grad():
        for y_ri, A_ri, H_ri in loader:
            y  = torch.view_as_complex(y_ri).to(device)
            A  = torch.view_as_complex(A_ri).to(device)
            Hs = torch.view_as_complex(H_ri).to(device)

            pred = model(y, A, Wr)
            loss = torch.mean(torch.norm(Hs - pred, dim=(-2,-1))**2 / torch.norm(Hs, dim=(-2,-1))**2)
            total_nmse += loss.item()

    avg_nmse = total_nmse / len(loader)
    nmse_db = 10 * np.log10(avg_nmse/T)
    print(f"Test NMSE = {nmse_db:.2f} dB")
    return nmse_db



def sweep_snr_plot(T=32, layers=10, snr_list=[-10, -5, 0, 5, 10]):
    results = []
    for snr in snr_list:
        checkpoint = f'checkpoints/RADMM_layers{layers}_nr132_NR232.pth'
        nmse_db = test_model(checkpoint, T, snr, layers)
        results.append(nmse_db)
    plt.figure()
    plt.plot(snr_list, results, marker='o')
    plt.xlabel('SNR (dB)')
    plt.ylabel('NMSE (dB)')
    plt.title(f'NMSE vs SNR (Layers={layers}, T={T})')
    plt.grid(True)
    plt.show()

def sweep_T_plot(SNR_dB=0, layers=10, T_list=[16, 32, 64]):
    results = []
    for T in T_list:
        checkpoint = f'checkpoints/RADMM_layers{layers}_nr132_NR232.pth'
        nmse_db = test_model(checkpoint, T, SNR_dB, layers)
        results.append(nmse_db)
    plt.figure()
    plt.plot(T_list, results, marker='o')
    plt.xlabel('Pilot Length T')
    plt.ylabel('NMSE (dB)')
    plt.title(f'NMSE vs Pilot Length (Layers={layers}, SNR={SNR_dB}dB)')
    plt.grid(True)
    plt.show()


def sweep_layers_plot(SNR_dB=0, T=32, layer_list=[1,2,3,4,5,6,7,8,9,10]):
    results = []
    for layers in layer_list:
        checkpoint = f'checkpoints/RADMM_layers{layers}_T{T}_SNR{SNR_dB}.pth'
        nmse_db = test_model(checkpoint, T, SNR_dB, layers)
        results.append(nmse_db)
    plt.figure()
    plt.plot(layer_list, results, marker='o')
    plt.xlabel('Number of Layers')
    plt.ylabel('NMSE (dB)')
    plt.title(f'NMSE vs Layers (SNR={SNR_dB}dB, T={T})')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    sweep_snr_plot()
    sweep_T_plot()
    sweep_layers_plot()
