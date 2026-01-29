import numpy as np
import matplotlib.pyplot as plt

def giatriham(x):
    return 30 + np.sum((x**2) - 10 * np.cos(2 * 3.14159265 * x))

def daoham(x):
    return 2 * x + 20 * 3.14159265 * np.sin(2 * 3.14159265 * x)

def momentum():
    x = np.random.uniform(-5.12, 5.12, 3)
    v = np.array([0.0, 0.0, 0.0]) # Để dạng float cho chắc
    a = 0.01    
    b = 0.9     
    t = 1000    
    i = 0 
    ls = []
    
    while i < t and daoham(x) > e:
        ls.append(giatriham(x)) 
        i = i + 1
        v_new = b*v + a * daoham(x)
        v = v_new
        x_moi = x - v
        x = x_moi
        
    fx = giatriham(x)
    return fx, ls 

def adagrad():
    x = np.random.uniform(-5.12, 5.12, 3)
    g = np.array([0.0, 0.0, 0.0]) 
    a = 0.01    
    e = 0.001    
    t = 1000    
    i = 0        
    
    ls = [] 
    
    while i < t and daoham(x) > e:
        ls.append(giatriham(x)) 
        i = i + 1
        g_new = g + (daoham(x)**2)
        g = g_new 
        x_moi = x - a / np.sqrt(g + e) * daoham(x)
        x = x_moi

    fx = giatriham(x)
    return fx, ls 

def adam():
    x = np.random.uniform(-5.12, 5.12, 3)
    g = np.array([0.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 0.0])
    a = 0.01    
    b = 0.9    
    p = 0.9  
    e = 0.001    
    t = 1000    
    i = 0       
    ls = []
    
    while i < t and daoham(x) > e:
        ls.append(giatriham(x)) 
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
    return fx, ls

bangmom = []
banggrad = []
bangadam = []

bdmom = []
bdgrad = []
bdadam = []

for k in range(20):
    kq, ls = momentum()
    bangmom.append(kq)
    bdmom = ls 

    kq, ls = adagrad()
    banggrad.append(kq)
    bdgrad = ls 
    
    kq, ls = adam()
    bangadam.append(kq)
    bdadam = ls

def bang_thong_ke(ten, bang):
    arr = np.array(bang)
    print(f"{ten:10} | Min: {np.min(arr):.4f} | Max: {np.max(arr):.4f} | Mean: {np.mean(arr):.4f} | Std: {np.std(arr):.4f}")

print("BẢNG THỐNG KÊ KẾT QUẢ")

bang_thong_ke("Momentum", bangmom)
bang_thong_ke("Adagrad", banggrad)
bang_thong_ke("Adam",     bangadam)

plt.figure(figsize=(10, 6))

plt.plot(bdmom[:1000], label='Momentum', color='blue')
plt.plot(bdgrad[:1000], label='Adagrad', color='green')
plt.plot(bdadam[:1000], label='Adam', color='red', linewidth=2)

plt.title(f'Biểu đồ quá trình hội tụ')
plt.xlabel('Số vòng lặp')
plt.ylabel('Giá trị hàm mục tiêu')
plt.legend()
plt.grid(True)
plt.savefig("bieu_do_hoi_tu.png")