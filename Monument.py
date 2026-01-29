import numpy as np

def giatriham(x):
    return 30 + np.sum((x**2) - 10 * np.cos(2 * 3.14159265 * x))

def daoham(x):
    return 2 * x + 20 * 3.14159265 * np.sin(2 * 3.14159265 * x)

def momentum():
    x = np.random.uniform(-5.12, 5.12, 3)
    v = np.zeros(3)
    a = 0.01    
    b = 0.9     
    t = 1000    
    i = 0 
    while i < t:
        v_new = b*v + a * daoham(x)
        v = v_new
        x_moi = x - v
        x = x_moi
        i = i + 1
    ket_qua = giatriham(x)
    return ket_qua

mangkq = []
for i in range(20):
    kq = momentum()
    mangkq.append(kq)
    print(f"{i+1}: {kq}")

mang_kq = np.array(mangkq)
min_val = np.min(mang_kq)
max_val = np.max(mang_kq)
mean_val = np.mean(mang_kq)
std_val = np.std(mang_kq)

print(f"Min: {min_val}")
print(f"Max: {max_val}")
print(f"Mean: {mean_val}")
print(f"Std: {std_val}")