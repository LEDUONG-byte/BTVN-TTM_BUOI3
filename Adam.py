import numpy as np

def giatriham(x):
    return 30 + np.sum((x**2) - 10 * np.cos(2 * 3.14159265 * x))

def daoham(x):
    return 2 * x + 20 * 3.14159265 * np.sin(2 * 3.14159265 * x)

def adam():
    x = np.random.uniform(-5.12, 5.12, 3)
    g = np.array([0, 0, 0])
    v = np.array([0, 0, 0])
    a = 0.01     
    b = 0.9     
    p = 0.9  
    e = 0.001     
    t = 1000     
    i = 0       
    
    while i < t:
        i = i + 1 
        v_new = b * v + (1 - b) * daoham(x)
        v = v_new

        g_new = p * g + (1 - p) * (daoham(x)**2)
        g = g_new

        v_mu = v / (1 - (b ** i))
        g_mu = g / (1 - (p ** i))

        x_moi = x - a * v_mu / (np.sqrt(g_mu) + e)
        x = x_moi

    fx = giatriham(x)
    return fx

mangkq = []
for k in range(20):
    kq = adam()
    mangkq.append(kq)
    print(f"{k+1}: {kq}")

mang_kq = np.array(mangkq)
min_val = np.min(mang_kq)
max_val = np.max(mang_kq)
mean_val = np.mean(mang_kq)
std_val = np.std(mang_kq)

print(f"Min: {min_val}")
print(f"Max: {max_val}")
print(f"Mean: {mean_val}")
print(f"Std: {std_val}")