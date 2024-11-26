# Data gangguan jaringan berdasarkan tabel yang diberikan
gangguan_data = [
    {"kode": "PG01", "nama": "Kabel Fiber Optic Rusak/Bermasalah", "gejala": ["GG01", "GG02", "GG03", "GG04"]},
    {"kode": "PG02", "nama": "Router Distribusi Mati/Bermasalah", "gejala": ["GG01", "GG02"]},
    {"kode": "PG03", "nama": "Modem Mati/Rusak", "gejala": ["GG01", "GG02", "GG04"]},
    {"kode": "PG04", "nama": "Kabel LAN dari Modem ke Router Tidak Tersambung/Bermasalah", "gejala": ["GG01", "GG02", "GG18", "GG22", "GG23"]},
    {"kode": "PG05", "nama": "Router Mati/Rusak", "gejala": ["GG01", "GG06", "GG08"]},
    {"kode": "PG06", "nama": "Kabel LAN Dari Router ke Switch Tidak Tersambung/Bermasalah", "gejala": ["GG01", "GG06", "GG18", "GG19", "GG20"]},
    {"kode": "PG07", "nama": "Switch Mati/Rusak", "gejala": ["GG01", "GG06", "GG08"]},
    {"kode": "PG08", "nama": "Perangkat Modem, Router, dan Switch Hang", "gejala": ["GG01", "GG06"]},
    {"kode": "PG09", "nama": "Looping", "gejala": ["GG21", "GG29", "GG30", "GG31"]},
    {"kode": "PG10", "nama": "Kabel LAN dari Switch ke Access Point Tidak Tersambung/Bermasalah", "gejala": ["GG11", "GG18", "GG24", "GG25", "GG26"]},
    {"kode": "PG11", "nama": "Perangkat Access Point Rusak/Mati", "gejala": ["GG11", "GG12", "GG13"]},
    {"kode": "PG12", "nama": "Perangkat Access Point Hang", "gejala": ["GG11", "GG14"]},
    {"kode": "PG13", "nama": "Collision", "gejala": ["GG29", "GG34", "GG35"]},
    {"kode": "PG14", "nama": "HUB Mati/Rusak", "gejala": ["GG06", "GG09", "GG10"]},
    {"kode": "PG15", "nama": "Kabel LAN dari HUB ruangan ke Komputer Tidak Tersambung/Bermasalah", "gejala": ["GG09", "GG15", "GG16", "GG17", "GG18"]},
    {"kode": "PG16", "nama": "LAN Card Komputer Rusak/Bermasalah", "gejala": ["GG09", "GG16", "GG27", "GG28"]},
    {"kode": "PG17", "nama": "Overload Bandwidth/Internet Limited Access", "gejala": ["GG21", "GG29", "GG30", "GG32"]},
    {"kode": "PG18", "nama": "Gangguan Massal (GAMAS) di Seluruh OPD", "gejala": ["GG01", "GG33"]}
]

# Fungsi untuk mencari jenis gangguan berdasarkan gejala yang diberikan
def cari_gangguan(evidence):
    hasil = []
    for gangguan in gangguan_data:
        # Periksa apakah semua gejala dalam data gangguan cocok dengan gejala yang diberikan
        if all(gejala in evidence for gejala in gangguan["gejala"]):
            hasil.append(f"{gangguan['kode']} - {gangguan['nama']} berdasarkan gejala: {', '.join(gangguan['gejala'])}")
    
    # Kembalikan hasil pencarian atau pesan jika tidak ditemukan
    if hasil:
        return "\n".join(hasil)
    else:
        return "Tidak ada jenis gangguan yang terdeteksi berdasarkan gejala yang diberikan."

# Meminta pengguna memasukkan gejala yang terdeteksi secara manual
print("Masukkan gejala yang terdeteksi, pisahkan dengan koma (contoh: GG01, GG02, GG03): ")
user_input = input("Gejala yang terdeteksi: ")
evidence = [gejala.strip() for gejala in user_input.split(",")]

# Panggil fungsi untuk mencari jenis gangguan
diagnosis = cari_gangguan(evidence)
print("\nHasil Pelacakan Gangguan:\n", diagnosis)
