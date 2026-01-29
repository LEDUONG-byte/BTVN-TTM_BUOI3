import numpy as np
import matplotlib.pyplot as plt

# --- 1. CÁC HÀM TÍNH TOÁN ---
def giatriham(x):
    return 30 + np.sum((x**2) - 10 * np.cos(2 * 3.14159265 * x))

def daoham(x):
    return 2 * x + 20 * 3.14159265 * np.sin(2 * 3.14159265 * x)

# --- 2. CÁC THUẬT TOÁN (Giữ nguyên logic của bạn, chỉ thêm phần lưu lịch sử) ---

def momentum():
    x = np.random.uniform(-5.12, 5.12, 3)
    v = np.array([0, 0, 0])
    a = 0.01    
    b = 0.9     
    t = 1000    
    i = 0 
    ls = []
    
    while i < t:
        ls.append(giatriham(x)) # <--- Lưu giá trị hiện tại vào lịch sử
        i = i + 1
        v_new = b*v + a * daoham(x)
        v = v_new
        x_moi = x - v
        x = x_moi
        
    fx = giatriham(x)
    return fx, ls # <--- Trả về thêm ls

def adagrad():
    x = np.random.uniform(-5.12, 5.12, 3)
    g = np.array([0, 0, 0]) 
    a = 0.01     
    e = 0.001     
    t = 1000     
    i = 0        
    
    ls = [] # <--- Thêm biến này
    
    while i < t:
        ls.append(giatriham(x)) # <--- Lưu giá trị
        i = i + 1
        g_new = g + (daoham(x)**2)
        g = g_new 
        x_moi = x - a / np.sqrt(g + e) * daoham(x)
        x = x_moi

    fx = giatriham(x)
    return fx, ls # <--- Trả về thêm ls

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
    
    ls = [] # <--- Thêm biến này
    
    while i < t:
        ls.append(giatriham(x)) # <--- Lưu giá trị
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
    return fx, ls # <--- Trả về thêm ls


# --- 3. CHẠY THỰC NGHIỆM ---

# Tạo 3 danh sách để chứa kết quả 20 lần chạy
kq_momentum = []; kq_adagrad = []; kq_adam = []

# Biến để lưu lại lịch sử của lần chạy cuối cùng
lich_su_mom = []; lich_su_ada = []; lich_su_adam = []

print("Đang chạy 20 lần cho mỗi thuật toán...")

for k in range(20):
    val, his = momentum() 
    kq_momentum.append(val)
    lich_su_mom = his 
    val, his = adagrad()
    kq_adagrad.append(val)
    lich_su_ada = his
    val, his = adam()
    kq_adam.append(val)
    lich_su_adam = his

# --- 4. IN BẢNG SO SÁNH (Stats) ---

def bang_thong_ke(ten, mang):
    arr = np.array(mang)
    print(f"{ten:10} | Min: {np.min(arr):.4f} | Max: {np.max(arr):.4f} | Mean: {np.mean(arr):.4f} | Std: {np.std(arr):.4f}")

print("BẢNG SO SÁNH CÁC CHỈ SỐ:")
bang_thong_ke("Momentum", kq_momentum)
bang_thong_ke("Adagrad", kq_adagrad)
bang_thong_ke("Adam", kq_adam)
