# 1
# import streamlit as st

# st.write("Hello world")

# 2
# import streamlit as st

# st.header('st.button')

# if st.button('hello baby'):
#     st.write('hello my baby')
# else:
#     st.write('Goodbye baby')

# 3
# import streamlit as st

# st.title("this is the app title")

# st.markdown("### this is the markdown")
# st.markdown("this is the header")
# st.markdown("## this is the subheader")
# st.caption("this is the caption")

# st.code("x = 2021", language="python")

# 4
# import streamlit as st

# if st.checkbox("yes"):
#     st.write("You selected Yes!")

# if st.button("Click"):
#     st.write("Button clicked!")

# st.write("Pick your gender")
# gender = st.radio("", ("Male", "Female"))
# st.write(f"You selected: {gender}")

# st.write("Pick your gender")
# selected_gender = st.selectbox("Select your gender", ["Male", "Female"])
# st.write(f"You picked: {selected_gender}")

# st.write("Choose a planet")
# planet = st.selectbox("Choose an option", ["Mercury", "Venus", "Earth", "Mars"])
# st.write(f"You selected: {planet}")

# st.write("Pick a mark")
# mark = st.slider("Bad ↔ Excellent", min_value=0, max_value=10, value=5)
# st.write(f"Mark: {mark}")

# st.write("Pick a number")
# number = st.slider("", min_value=0, max_value=50, value=25)
# st.write(f"Number: {number}")

# 5
# import streamlit as st

# st.number_input("Pick a number", min_value=0, max_value=100, step=1)

# st.text_input("Email address")

# st.date_input("Travelling date")

# st.time_input("School time")

# st.text_area("Description")

# st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])

# st.color_picker("Choose your favourite color", "#ff00ff")

# 6
# import numpy as np
# import altair as alt
# import pandas as pd
# import streamlit as st

# st.header('st.write Example')

# st.write('Hello, World! :sunglasses:')

# st.write(1234)

# df = pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
# })
# st.write(df)

# st.write('Below is a DataFrame:', df, 'Above is a DataFrame.')

# df2 = pd.DataFrame(
#     np.random.randn(200, 3),
#     columns=['a', 'b', 'c']
# )

# c = alt.Chart(df2).mark_circle().encode(
#     x='a',
#     y='b',
#     size='c',
#     color='c',
#     tooltip=['a', 'b', 'c']
# )

# st.write(c)

# 7
# import streamlit as st
# import pandas as pd
# import numpy as np

# df = pd.DataFrame(
#     np.random.randn(20, 2),
#     columns=['x', 'y']
# )
# st.line_chart(df)

# chart_data = pd.DataFrame(
#     np.random.randn(20, 2),
#     columns=["x", "y",]
# )
# st.bar_chart(chart_data)

# chart_data = pd.DataFrame(
#     np.random.randn(20, 2),
#     columns=["a", "b",]
#     )
# st.area_chart(chart_data)

# 8
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# page = st.sidebar.selectbox("Select Page", ["Home"])

# if page == "Home":
#     st.title("BANK MARKETING")
#     st.image(r"C:\sistem cerdas\FuzzyLogic\myproject\bankku.jpg", caption="Image Example", use_container_width=True)

#     st.subheader("Dataset Preview")

#     df = pd.read_csv(r"C:\sistem cerdas\FuzzyLogic\myproject\bank-additional-full.csv")
#     st.write(df.head())

#     if 'Job Title' in df.columns and 'Salary' in df.columns:
#         st.subheader("Bar Chart of Average Salary by Job Title")
        
#         salary_by_job_title = df.groupby('Job Title')['Salary'].mean().sort_values(ascending=False)
        
#         fig, ax = plt.subplots()
#         salary_by_job_title.plot(kind='bar', ax=ax, color='skyblue')
#         ax.set_title('Average Salary by Job Title')
#         ax.set_xlabel('Job Title')
#         ax.set_ylabel('Average Salary')
#         ax.tick_params(axis='x', rotation=45)
#         st.pyplot(fig)

#     st.subheader("Random Data Chart")

#     df_chart = pd.DataFrame(np.random.randn(100, 2), columns=["x", "y"])

#     chart_option = st.selectbox("Select Chart Type", ["Line Chart", "Bar Chart", "Area Chart"])

#     if chart_option == "Line Chart":
#         st.line_chart(df_chart)

#     elif chart_option == "Bar Chart":
#         st.bar_chart(df_chart)

#     elif chart_option == "Area Chart":
#         st.area_chart(df_chart)

# 9 - 10
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import pickle
# import os
# import streamlit as st

# if not os.path.exists('CarPrice_Assignment.csv'):
#     st.write("File 'CarPrice_Assignment.csv' tidak ditemukan!")
# else:
#     df_mobil = pd.read_csv('CarPrice_Assignment.csv')

#     descriptive_stats = df_mobil.describe()
#     st.write("Descriptive Statistics:")
#     st.write(descriptive_stats)

#     st.write("Data Types:")
#     st.write(df_mobil.dtypes)

#     st.write("Distribusi Harga Mobil:")
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.title('Car Distribution Plot')
#     sns.histplot(df_mobil['price'])
#     st.pyplot(plt)

#     car_counts = df_mobil['CarName'].value_counts()
#     st.write("Distribusi Jumlah Mobil Berdasarkan CarName:")
#     plt.figure(figsize=(10, 6))
#     car_counts.plot(kind="bar")
#     plt.title("CarName Distribution")
#     plt.xlabel("CarName")
#     plt.ylabel("Count")
#     plt.xticks(rotation=45)
#     st.pyplot(plt)

#     top_10_cars = df_mobil['CarName'].value_counts().head(10)
#     st.write("10 Nama Mobil Terbanyak:")
#     st.write(top_10_cars)

#     st.write("Visualisasi 10 Nama Mobil Terbanyak:")
#     plt.figure(figsize=(10, 6))
#     car_counts.head(10).plot(kind="bar", color="blue")
#     plt.title("10 Nama Mobil Terbanyak pada Dataset", fontsize=16)
#     plt.xlabel("Nama Mobil", fontsize=12)
#     plt.ylabel("Jumlah", fontsize=12)
#     plt.xticks(rotation=45, fontsize=10)
#     plt.tight_layout()
#     st.pyplot(plt)

#     car_names = " ".join(df_mobil['CarName'])
#     wordcloud = WordCloud(
#         width=800, height=400,
#         background_color='white',
#         colormap='viridis',
#         random_state=42
#     ).generate(car_names)

#     st.write("Word Cloud of Car Names:")
#     plt.figure(figsize=(12, 6))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.title("Word Cloud of Car Names", fontsize=16)
#     st.pyplot(plt)

#     st.write("Scatter Plot Harga Mobil vs Highwaympg:")
#     plt.scatter(df_mobil['highwaympg'], df_mobil['price'])
#     plt.xlabel('highwaympg')
#     plt.ylabel('price')
#     st.pyplot(plt)

#     x = df_mobil[['highwaympg', 'curbweight', 'horsepower']]
#     y = df_mobil['price']

#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#     model_regresi = LinearRegression()
#     model_regresi.fit(x_train, y_train)

#     model_regresi_pred = model_regresi.predict(x_test)

#     st.write("Prediksi vs Harga Sebenarnya:")
#     plt.scatter(x_test.iloc[:, 0], y_test, label='Actual Price', color='blue')
#     plt.scatter(x_test.iloc[:, 0], model_regresi_pred, label='Predicted Prices', color='red')
#     plt.xlabel('highwaympg')
#     plt.ylabel('price')
#     plt.legend()
#     st.pyplot(plt)

#     X = np.array([[32, 2338, 75]])
#     harga_X = model_regresi.predict(X)
#     harga_X_int = int(harga_X[0])
#     st.write(f'Harga Prediksi untuk Input: {harga_X_int}')

#     mae = mean_absolute_error(y_test, model_regresi_pred)
#     st.write(f'Mean Absolute Error (MAE): {mae:.2f}')

#     mse = mean_squared_error(y_test, model_regresi_pred)
#     st.write(f'Mean Square Error (MSE): {mae:.2f}')

#     rmse = np.sqrt(mse)
#     st.write(f'Root Mean Square Error (RMSE): {rmse:.2f}')

#     filename = 'model_prediksi_harga_mobil.sav'
#     pickle.dump(model_regresi, open(filename, 'wb'))
#     st.write("Model berhasil disimpan!")

# 11
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os
import streamlit as st

# Judul Aplikasi
st.title("Analisis dan Prediksi Harga Mobil")
st.markdown("""
Aplikasi ini memvisualisasikan data mobil dan melakukan prediksi harga berdasarkan fitur-fitur tertentu.
""")

# Memeriksa keberadaan file dataset
if not os.path.exists('CarPrice_Assignment.csv'):
    st.error("File **'CarPrice_Assignment.csv'** tidak ditemukan! Pastikan file berada di direktori yang sesuai.")
else:
    # Membaca dataset
    df_mobil = pd.read_csv('CarPrice_Assignment.csv')
    st.sidebar.header("Info Dataset")
    st.sidebar.write(f"Jumlah Baris: {df_mobil.shape[0]}")
    st.sidebar.write(f"Jumlah Kolom: {df_mobil.shape[1]}")
    
    # 1. Deskripsi Data
    st.subheader("1. Deskripsi Data")
    with st.expander("Statistik Deskriptif"):
        st.write(df_mobil.describe())

    with st.expander("Tipe Data Kolom"):
        st.write(df_mobil.dtypes)

    # 2. Visualisasi Data
    st.subheader("2. Visualisasi Data")
    
    # Visualisasi Distribusi Harga Mobil
    st.markdown("### Distribusi Harga Mobil")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df_mobil['price'], kde=True, color='green', ax=ax)
    ax.set_title("Distribusi Harga Mobil", fontsize=14)
    st.pyplot(fig)
    
    # Visualisasi Distribusi Jumlah Mobil Berdasarkan Nama
    st.markdown("### Distribusi Jumlah Mobil Berdasarkan CarName")
    top_10_cars = df_mobil['CarName'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    top_10_cars.plot(kind="bar", color="skyblue", ax=ax)
    ax.set_title("10 Nama Mobil Terbanyak", fontsize=14)
    ax.set_xlabel("CarName")
    ax.set_ylabel("Jumlah")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Visualisasi Word Cloud dari Nama Mobil
    st.markdown("### Word Cloud Nama Mobil")
    car_names = " ".join(df_mobil['CarName'])
    wordcloud = WordCloud(
        width=800, height=400, background_color='white', colormap='viridis'
    ).generate(car_names)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    
    # Scatter Plot Harga Mobil vs Highwaympg
    st.markdown("### Scatter Plot: Harga Mobil vs Highwaympg")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df_mobil['highwaympg'], y=df_mobil['price'], ax=ax)
    ax.set_title("Harga Mobil vs Highwaympg", fontsize=14)
    ax.set_xlabel("Highway MPG")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    # 3. Model Prediksi
    st.subheader("3. Prediksi Harga Mobil")
    
    # Persiapan data untuk model
    x = df_mobil[['highwaympg', 'curbweight', 'horsepower']]
    y = df_mobil['price']

    # Membagi data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Membuat model regresi linear
    model_regresi = LinearRegression()
    model_regresi.fit(x_train, y_train)

    # Prediksi data uji
    model_regresi_pred = model_regresi.predict(x_test)

    # Visualisasi Prediksi vs Aktual dengan Select Slider
    st.markdown("### Prediksi vs Harga Sebenarnya")
    st.markdown("Pilih atribut untuk divisualisasikan:")
    selected_feature = st.select_slider(
        "Pilih atribut:",
        options=["highwaympg", "curbweight", "horsepower"],
        value="highwaympg"
    )

    # Visualisasi dinamis berdasarkan fitur yang dipilih
    st.write(f"Menampilkan grafik untuk atribut: **{selected_feature}**")
    fig, ax = plt.subplots()
    ax.scatter(x_test[selected_feature], y_test, label="Harga Sebenarnya", color="blue", alpha=0.7)
    ax.scatter(x_test[selected_feature], model_regresi_pred, label="Harga Prediksi", color="red", alpha=0.7)
    ax.set_title(f"Harga Sebenarnya vs Prediksi ({selected_feature.capitalize()})", fontsize=14)
    ax.set_xlabel(selected_feature.capitalize())
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # Prediksi harga untuk input baru
    st.markdown("### Prediksi untuk Input Baru")
    X_new = np.array([[32, 2338, 75]])
    prediksi_harga = model_regresi.predict(X_new)
    st.success(f"Harga Prediksi untuk Input Baru: **${int(prediksi_harga[0]):,}**")

    # Menampilkan Error
    mae = mean_absolute_error(y_test, model_regresi_pred)
    mse = mean_squared_error(y_test, model_regresi_pred)
    rmse = np.sqrt(mse)
    st.markdown("### Evaluasi Model")
    st.write(f"- **Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"- **Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"- **Root Mean Squared Error (RMSE):** {rmse:.2f}")

    # Menyimpan model
    filename = 'model_prediksi_harga_mobil.sav'
    pickle.dump(model_regresi, open(filename, 'wb'))
    st.success("Model berhasil disimpan sebagai 'model_prediksi_harga_mobil.sav'!")
