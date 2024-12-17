import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from streamlit_option_menu import option_menu
import plotly.express as px 
from datetime import datetime
from sklearn.cluster import KMeans
import calendar
import streamlit as st
from sklearn.preprocessing import StandardScaler
st.set_page_config(layout="wide")

# ----------------------------
# 1. Data Preprocessing
# ----------------------------

@st.cache_data
def load_and_preprocess_data(uploaded_file, today_str=None):
    if today_str is None:
        today_str = datetime.now().strftime('%Y-%m-%d')
    if uploaded_file is None:
        st.warning("Silakan unggah file CSV terlebih dahulu.")
        st.stop()
    try:
        data = pd.read_csv(uploaded_file)
        st.success("File CSV berhasil diunggah dan dimuat.")
    except pd.errors.EmptyDataError:
        st.error("File CSV yang diunggah kosong.")
        st.stop()
    except pd.errors.ParserError:
        st.error("Terjadi kesalahan saat mengurai file CSV. Pastikan formatnya benar.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.stop()

    if 'Date' not in data.columns:
        st.error("Kolom 'Date' tidak ditemukan dalam file CSV.")
        st.stop()

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    if data['Date'].isnull().any():
        st.warning("Beberapa entri 'Date' tidak dapat diubah menjadi format datetime dan diatur sebagai NaT.")

    today = pd.Timestamp(today_str)
    # Memfilter data dengan status 'SUCCESS' dan Total Transaksi bukan 0
    data= data[(data['Status'] == 'SUCCESS') & (data['Total Transaksi'] != 0)]

    # Ubah semua nilai di kolom 'Donatur' menjadi huruf kecil
    data['Donatur'] = data['Donatur'].str.lower()

    # Mengisi missing value Campaign dengan mode berdasarkan Program
    data = isi_missing_campaign(data)

    columns = data.columns.tolist()
    #columns.insert(columns.index('Date'), columns.pop(columns.index('id_transaksi')))
    data = data[columns]

    if 'Donatur' not in data.columns:
        st.error("Kolom 'Donatur' tidak ditemukan dalam file CSV.")
        st.stop()
    data = data.sort_values(by=['Date'],ascending=False)

    if 'selisih_hari' not in data.columns:
        # Urutkan data per Donatur berdasarkan Date
        data = data.sort_values(by=['Donatur', 'Date'], ascending=[True, True])

        # Hitung selisih hari dengan diff(-1) untuk mendapatkan selisih maju
        data['selisih_hari'] = data.groupby('Donatur')['Date'].diff(-1).dt.days * -1
    #data['selisih_hari'] = data['selisih_hari'].replace(-0.0, 0)
    data['recency'] = (today - data['Date']).dt.days
    #data = data.dropna(subset=['selisih_hari'])

    
    return data

# Fungsi untuk mengisi missing value dengan mode berdasarkan program
def isi_missing_campaign(df_success):
    df_success['Campaign'] = df_success.groupby('Program')['Campaign'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else '-'))
    if df_success['Campaign'].isna().sum() > 0:
        mode_seluruh_campaign = df_success['Campaign'].mode()[0]  # Mengambil mode dari seluruh kolom Campaign
        df_success['Campaign'].fillna(mode_seluruh_campaign, inplace=True)
    return df_success

# Fungsi untuk mendapatkan tren donasi donatur
def get_donor_trend(data, donor_name, selected_year):
    data['Year'] = data['Date'].dt.year
    donor_data = data[(data['Donatur'].str.lower() == donor_name) & (data['Year'] == selected_year)]
    if donor_data.empty:
        return donor_data
    donor_data['Year-Month'] = donor_data['Date'].dt.to_period('M').astype(str)
    trend_data = donor_data.groupby('Year-Month')['Total Transaksi'].sum().reset_index()
    trend_data['Month'] = trend_data['Year-Month'].apply(lambda x: calendar.month_name[int(x.split('-')[1])])
    return trend_data

# Fungsi untuk format rupiah
def format_rupiah(value):
    """Format angka menjadi Rupiah dengan separator titik."""
    return f"{value:,.0f}".replace(",", ".")

# ----------------------------
# 2. Streamlit App Layout
# ----------------------------

# Sidebar
with st.sidebar:
    st.image("wakaf-logo.png", use_column_width=True)
    selected = option_menu(
        menu_title=None,
        options=["Upload File", "RFM", "Tren"],
        icons=["upload", "bar-chart","graph-up"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#f8f9fa","color":"#0E1117"},
            "icon": {"color": "#0E1117", "font-size": "20px","--hover-color": "#f8f9fa"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#DBD3D3","color":"#0E1117"},
            "nav-link-selected": {"background-color": "#0f0f43","color":"#f8f9fa"},
        }
)

# Fitur Upload File
if selected == "Upload File":
    st.header("Unggah File CSV Donasi")
    uploaded_file = st.file_uploader("Pilih file CSV Anda", type=['csv'])
    if uploaded_file is not None:
        data = load_and_preprocess_data(uploaded_file)
        st.session_state.data = data
        if data is not None:
            st.subheader("Data Setelah Preprocessing:")
            st.write(data)

# Fitur RFM
elif selected == "RFM":
        st.header("RFM Analysis")    
        if 'data' not in st.session_state:
            st.error('Silakan unggah file CSV terlebih dahulu.')
        else:
            data = st.session_state.data

            reselected_columns = ['Donatur', 'Program', 'Total Transaksi', 'Date']
            df_rfm = data[reselected_columns]

            df_rfm['Date'] = pd.to_datetime(df_rfm['Date'], format='%Y-%m-%d', errors='coerce')

            today = pd.Timestamp.now()

            df_rfm_agg = df_rfm.groupby('Donatur').agg(
                donasi_terakhir=('Date', 'max'),               # Ambil tanggal donasi terakhir
                frequency=('Donatur', 'count'),               # Hitung frekuensi donasi
                monetary=('Total Transaksi', 'sum')                # Jumlahkan total transaksi
            ).reset_index()

            # Hitung recency berdasarkan hari terakhir transaksi dari hari ini
            df_rfm_agg['recency'] = (today - df_rfm_agg['donasi_terakhir']).dt.days

            # Format monetary
            df_rfm_agg['monetary'] = df_rfm_agg['monetary'].round()

            # Skoring RFM
            def r_score(x):
                return 5 if x <= 90 else 4 if x <= 180 else 3 if x <= 270 else 2 if x <= 365 else 1
            def f_score(x):
                return 5 if x > 12 else 4 if x >= 10 else 3 if x >= 7 else 2 if x >= 3 else 1
            def m_score(x):
                return 5 if x > 10000000 else 4 if x >= 5000001 else 3 if x >= 1000001 else 2 if x >= 500001 else 1

            # Menambahkan RFM Skor
            df_rfm_agg['r_score'] = df_rfm_agg['recency'].apply(r_score)
            df_rfm_agg['f_score'] = df_rfm_agg['frequency'].apply(f_score)
            df_rfm_agg['m_score'] = df_rfm_agg['monetary'].apply(m_score)

           # Menghapus donatur dengan nilai RFM yang tidak terdefinisi
            df_rfm_agg = df_rfm_agg.dropna(subset=['r_score', 'f_score', 'm_score'])

            # Pastikan skor RFM adalah integer
            df_rfm_agg['r_score'] = df_rfm_agg['r_score'].astype(int)
            df_rfm_agg['f_score'] = df_rfm_agg['f_score'].astype(int)
            df_rfm_agg['m_score'] = df_rfm_agg['m_score'].astype(int)

            # Melalukan pelabelan
            def assign_labels(cluster):
                if cluster == 0:
                    return "Muwakif Rentan"
                elif cluster == 1:
                    return "Muwakif Potensial"
                elif cluster == 2:
                    return "Muwakif Loyal"
                elif cluster == 3:
                    return "Muwakif Terbaik"
                else:
                    return "Tidak ada kategori"  # If any other cluster is created
                
            # Scale RFM scores
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(df_rfm_agg[['r_score', 'f_score', 'm_score']])

            optimal_clusters = 4  # Hasil elbow method menunjukkan 4 cluster
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            df_rfm_agg['Cluster'] = kmeans.fit_predict(rfm_scaled)

            df_rfm_agg['Kategori'] = df_rfm_agg['Cluster'].apply(assign_labels)
            
            df_rfm_agg['Kategori'] = df_rfm_agg['Kategori'].fillna('Donatur Belum Diketahui')
    
            unknown_donors = df_rfm_agg[df_rfm_agg['Kategori'] == 'Donatur Belum Diketahui']
            
            sorted_df = df_rfm_agg.sort_values(by='donasi_terakhir', ascending=False)
           
            st.session_state.sorted_df = sorted_df

            final_rfm = pd.merge(data, df_rfm_agg[['Donatur', 'Kategori']], on="Donatur")

            st.session_state.final_rfm = final_rfm

            final_rfm['Date'] = pd.to_datetime(final_rfm['Date'])
            final_rfm['year'] = final_rfm['Date'].dt.year  

            # Statistik
            total_donatur = final_rfm['Donatur'].nunique()
            total_donasi_success = final_rfm['Total Transaksi'].sum()
            average_donation = final_rfm['Total Transaksi'].mean()


            st.subheader("Statistik Donasi")
            viz_col0, control_col0 = st.columns([3, 1])
            with control_col0:
                # Filter berdasarkan kategori
                categories = final_rfm['Kategori'].unique() 

                selected_Kategori_treemap = st.selectbox("Pilih Kategori", categories)
                filtered_data = final_rfm[final_rfm['Kategori'] == selected_Kategori_treemap]

                total_donatur = filtered_data['Donatur'].nunique()
                total_donasi_success = filtered_data['Total Transaksi'].sum()

                best_program = filtered_data.groupby('Program')['Total Transaksi'].sum().idxmax()            
                best_campaign = filtered_data.groupby('Campaign')['Total Transaksi'].sum().idxmax()

            with viz_col0:
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    st.metric("Total Donatur", total_donatur, "Orang")
                with stat_col2:
                    st.metric("Total Donasi", f"Rp {total_donasi_success:,.2f}", "Donasi Berhasil")
                st.markdown("<hr>", unsafe_allow_html=True)

            viz_col1, control_col1 = st.columns([3, 1])
            with viz_col1:
                prog_col1, prog_col2 = st.columns(2)

                with prog_col1:
                    st.markdown("Program Terbaik")
                    st.markdown(f'<div style="font-size: 32px;">{best_program}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="color: #09ab3b;">Sampai saat ini</div>', unsafe_allow_html=True)

                with prog_col2:
                    st.markdown("Campaign Terbaik")
                    st.markdown(f'<div style="font-size: 32px;">{best_campaign}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="color: #09ab3b;">Sampai saat ini</div>', unsafe_allow_html=True)
            
            with open('donor_classification.json', 'r') as file:
                donor_classification = json.load(file)  
            st.markdown("<hr>", unsafe_allow_html=True)

            st.subheader("Pelayanan Donatur")
            donor_name = st.text_input("Masukan Nama Donatur Untuk Melihat Kategori:", "").strip().lower()
            
            if donor_name:
                donor_name = donor_name.lower()
                donor_data = df_rfm_agg[df_rfm_agg['Donatur'].str.lower() == donor_name]
                
                if not donor_data.empty:
                    donor_category = donor_data['Kategori'].values[0] 
                    donor_characteristic = donor_data['Karakteristik'].values[0] if 'Karakteristik' in donor_data else 'Tidak ada informasi'  # Ambil karakteristik jika ada
                    donor_r=donor_data['recency'].values[0]
                    donor_f=donor_data['frequency'].values[0]
                    donor_m=donor_data['monetary'].values[0]
                    
                    classification_data = next((item for item in donor_classification if item['kategori'] == donor_category), None)
                    if donor_category is None or donor_category == 'Donatur Belum Diketahui':
                        
                        st.warning("Mohon Maaf Donatur Tersebut Kategori DONATUR BELUM DI KETAHUI (Belum Terdefinisi)")
                    elif classification_data:
                        
                        donor_characteristic=classification_data['karakteristik']
                        donor_strategy = classification_data['strategi_pelayanan']  

                        st.success(f"Nama {donor_name.upper()} adalah kategori {donor_category.upper()}.")
                        st.markdown(f'''Karakteristik : :green[{donor_characteristic.upper()}]''')
                        st.markdown(f'''Strategi Pelayanan : :green[{donor_strategy.upper()}]''')
                        st.write(f"**Terakhir berdonasi**: {donor_r} HARI LALU")
                        st.write(f"**Banyak Berdonasi**: {donor_f} KALI")
                        donor_m_formatted = f"Rp {donor_m:,.0f}"  
                        st.write(f"**Total Seluruh Donasi**: {donor_m_formatted}")  
                else:
                    st.warning(f"Donatur dengan nama '{donor_name}' tidak ditemukan.")
            st.markdown("<hr>", unsafe_allow_html=True)

            st.subheader("Tabel Data")
            option = st.selectbox("Pilih Tabel untuk Ditampilkan:", ("Tidak Ada", "Hasil RFM Analysis", "Hasil Final RFM"))
            if option == "Hasil RFM Analysis":
                st.subheader("Hasil RFM Analysis:")
                st.write(sorted_df)
            elif option == "Hasil Final RFM":
                st.subheader("Hasil Final RFM:")
                st.write(final_rfm)
                csv = sorted_df.to_csv(index=False).encode('utf-8')
                st.download_button("Unduh Hasil RFM", csv, "rfm_analysis.csv", "text/csv", key='download-csv')
            st.markdown("<hr>", unsafe_allow_html=True)

            st.subheader("Visualisasi Data")
            viz_col2, control_col2 = st.columns([3, 1])

            with viz_col2:
                if 'selected_Kategori_bar' in st.session_state:
                    selected_Kategori_bar = st.session_state['selected_Kategori_bar']

                    if selected_Kategori_bar == 'Kategori':
                        df_rfm_agg = sorted_df.groupby('Kategori').size().reset_index(name='Jumlah Donatur')

                        max_donatur_row = df_rfm_agg.loc[df_rfm_agg['Jumlah Donatur'].idxmax()]
                        min_donatur_row = df_rfm_agg.loc[df_rfm_agg['Jumlah Donatur'].idxmin()]

                        max_kategori = max_donatur_row['Kategori']
                        max_donatur = max_donatur_row['Jumlah Donatur']

                        min_kategori = min_donatur_row['Kategori']
                        min_donatur = min_donatur_row['Jumlah Donatur']

                        trend_text = (
                            f"Kategori <span style='color: green;'><b>{max_kategori}</b></span> "
                            f"memiliki jumlah donatur tertinggi dengan total donatur sebesar "
                            f"<span style='color: green;'><b>{max_donatur}</b></span>, "
                            f"sementara kategori <span style='color: red;'><b>{min_kategori}</b></span> "
                            f"memiliki jumlah donatur terendah dengan total donatur sebesar "
                            f"<span style='color: red;'><b>{min_donatur}</b></span>."
                        )

                        fig = px.bar(
                            df_rfm_agg,
                            x='Kategori',
                            y='Jumlah Donatur',
                            title="Distribusi Jumlah Donatur Berdasarkan Kategori",
                            labels={'Jumlah Donatur': 'Jumlah Donatur', 'Kategori': 'Kategori'}
                        )
                       
                    elif selected_Kategori_bar == 'Monetary':
                        df_rfm_agg = sorted_df.groupby('Kategori')['monetary'].sum().reset_index()

                        max_monetary_row = df_rfm_agg.loc[df_rfm_agg['monetary'].idxmax()]
                        min_monetary_row = df_rfm_agg.loc[df_rfm_agg['monetary'].idxmin()]

                        max_kategori = max_monetary_row['Kategori']
                        max_monetary = max_monetary_row['monetary']

                        min_kategori = min_monetary_row['Kategori']
                        min_monetary = min_monetary_row['monetary']

                        trend_text = (
                            f"Kategori <span style='color: green;'><b>{max_kategori}</b></span> "
                            f"memiliki total monetary tertinggi dengan jumlah total "
                            f"<span style='color: green;'><b>{format_rupiah(max_monetary)}</b></span>, "
                            f"sementara kategori <span style='color: red;'><b>{min_kategori}</b></span> "
                            f"memiliki total monetary terendah dengan jumlah total "
                            f"<span style='color: red;'><b>{format_rupiah(min_monetary)}</b></span>."
                        )

                        fig = px.bar(
                            df_rfm_agg,
                            x='Kategori',
                            y='monetary',
                            title="Distribusi Total Monetary Berdasarkan Kategori",
                            labels={'monetary': 'Total Monetary (IDR)', 'Kategori': 'Kategori'}
                        )

                        fig.update_layout(
                            yaxis_tickformat=",.0f",  
                            yaxis_tickprefix="Rp "  
                        )

                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(trend_text, unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            with control_col2:
                st.header("Kontrol Bar Chart")
                Kategori_options = ['Kategori', 'Monetary']
                selected_Kategori_bar = st.selectbox("Pilih Tipe Visualisasi:", Kategori_options, key='selected_Kategori_bar')

            viz_col3, control_col3 = st.columns([3, 1])

            with viz_col3:
                if 'selected_Kategori_treemap' in st.session_state:
                    selected_Kategori_treemap = st.session_state['selected_Kategori_treemap']

                    treemap_data = final_rfm.groupby(selected_Kategori_treemap)['Donatur'].nunique().reset_index(name='count')
                    total_count = treemap_data['count'].sum()
                    treemap_data['percentage'] = (treemap_data['count'] / total_count * 100).round(2)

                    max_donatur_row = treemap_data.loc[treemap_data['count'].idxmax()]
                    min_donatur_row = treemap_data.loc[treemap_data['count'].idxmin()]

                    max_kategori = max_donatur_row[selected_Kategori_treemap]
                    max_donatur = max_donatur_row['count']
                    max_percentage = max_donatur_row['percentage']

                    min_kategori = min_donatur_row[selected_Kategori_treemap]
                    min_donatur = min_donatur_row['count']
                    min_percentage = min_donatur_row['percentage']

                    trend_text = (
                        f"Kategori <span style='color: green;'><b>{max_kategori}</b></span> "
                        f"memiliki jumlah donatur tertinggi dengan total donatur sebanyak "
                        f"<span style='color: green;'><b>{max_donatur}</b></span> "
                        f"({max_percentage}% dari total donatur), "
                        f"sementara kategori <span style='color: red;'><b>{min_kategori}</b></span> "
                        f"memiliki jumlah donatur terendah dengan total donatur sebanyak "
                        f"<span style='color: red;'><b>{min_donatur}</b></span> "
                        f"({min_percentage}% dari total donatur)."
                    )
                    
                    fig = px.treemap(
                        treemap_data, 
                        path=[selected_Kategori_treemap], 
                        values='count', 
                        title=f'Segmentasi Donatur berdasarkan {selected_Kategori_treemap}',
                        color='count',
                        hover_data={'count': True, 'percentage': True}
                    )
                    fig.update_traces(
                        hovertemplate="<b>%{label}</b><br>Total: %{value}<br>Persentase: %{customdata[1]}%"
                    )
                    st.plotly_chart(fig)
                    st.markdown(trend_text, unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)

            with control_col3:
                st.header("Kontrol Treemap")
                Kategori_options_treemap = ['Campaign', 'Program']  # Tambahkan kategori lain jika perlu
                selected_Kategori_treemap = st.selectbox("Pilih Kategori untuk Treemap:", Kategori_options_treemap, key='selected_Kategori_treemap')

            viz_col4, control_col4 = st.columns([3, 1])
            with control_col4:
                st.header("Kontrol Tren Donasi")
                final_rfm['Year'] = final_rfm['Date'].dt.year
                unique_years = sorted(final_rfm['Year'].unique())
                selected_year = st.selectbox("Pilih Tahun:", unique_years, key='selected_year')
            with viz_col4:
                filtered_data = final_rfm[final_rfm['Year'] == selected_year]
                filtered_data['Year-Month'] = filtered_data['Date'].dt.to_period('M').astype(str)
                monthly_donations = filtered_data.groupby('Year-Month')['Total Transaksi'].sum().reset_index()

                max_donation_row = monthly_donations.loc[monthly_donations['Total Transaksi'].idxmax()]
                min_donation_row = monthly_donations.loc[monthly_donations['Total Transaksi'].idxmin()]

                max_month = max_donation_row['Year-Month']
                max_donation = max_donation_row['Total Transaksi']

                min_month = min_donation_row['Year-Month']
                min_donation = min_donation_row['Total Transaksi']

                max_month_name = calendar.month_name[int(max_month.split('-')[1])]
                min_month_name = calendar.month_name[int(min_month.split('-')[1])]

                trend_text = (
                    f"Pada tahun {selected_year}, bulan <span style='color: green;'><b>{max_month_name}</b></span> "
                    f"mengalami donasi tertinggi dengan total donasi sebesar <span style='color: green;'><b>{format_rupiah(max_donation)}</b></span>, "
                    f"sementara bulan <span style='color: red;'><b>{min_month_name}</b></span> mengalami donasi terendah dengan total donasi sebesar "
                    f"<span style='color: red;'><b>{format_rupiah(min_donation)}</b></span>."
                )

                fig_trend = px.line(
                    monthly_donations,
                    x='Year-Month',
                    y='Total Transaksi',
                    title=f"Tren Donasi Bulanan Tahun {selected_year}",
                    labels={'Year-Month': 'Bulan', 'Total Transaksi': 'Total Donasi (IDR)'},
                    markers=True
                )

                fig_trend.update_yaxes(
                    tickformat=".0f",  
                    tickprefix="Rp "   
                )

                fig_trend.update_traces(
                    hovertemplate="<b>%{x}</b><br>Total Donasi: Rp %{y:,.0f}<br>"
                )

                st.plotly_chart(fig_trend, use_container_width=True)
                st.markdown(trend_text, unsafe_allow_html=True)

elif selected == "Tren":
    st.header("Tren Donasi Donatur")

    if 'data' not in st.session_state:
        st.error('Silakan unggah file CSV terlebih dahulu.')
    else:
        if 'final_rfm' not in st.session_state:
            st.error('Silakan lakukan analisis RFM terlebih dahulu.')
        else:
            st.subheader("Visualisasi Tren Donasi")
            data_trend = st.session_state.final_rfm

            data_trend['Year'] = data_trend['Date'].dt.year
            unique_years = sorted(data_trend['Year'].unique())

            viz_col0, control_col0 = st.columns([3, 1])
            with control_col0:
                st.header("Kontrol Tren Donasi")
                selected_year = st.selectbox("Pilih Tahun", unique_years)
            with viz_col0:
                donor_name = st.text_input('Masukkan Nama Donatur').lower()
                if donor_name:
                    trend_data = get_donor_trend(data_trend, donor_name, selected_year)
                    if trend_data.empty:
                        st.error(f'Tidak ada data untuk donatur dengan nama: {donor_name.upper()} dan tahun {selected_year}.')
                    else:
                        fig_trend = px.line(
                            trend_data,
                            x='Month',
                            y='Total Transaksi',
                            title=f"Tren Donasi {donor_name.upper()} Pada Tahun {selected_year}",
                            labels={'Month': 'Bulan', 'Total Transaksi': 'Total Donasi (IDR)'},
                            markers=True
                        )

                        fig_trend.update_yaxes(
                            tickformat=".0f",  # Menghilangkan desimal
                            tickprefix="Rp "   # Menambahkan prefix "Rp"
                        )

                        fig_trend.update_traces(
                            hovertemplate='%{x}: %{y:,.0f} <extra></extra>'
                        )

                        st.plotly_chart(fig_trend)
                        
                        max_donation_row = trend_data.loc[trend_data['Total Transaksi'].idxmax()]
                        min_donation_row = trend_data.loc[trend_data['Total Transaksi'].idxmin()]

                        max_month = max_donation_row['Month']
                        max_donation = max_donation_row['Total Transaksi']

                        min_month = min_donation_row['Month']
                        min_donation = min_donation_row['Total Transaksi']

                        trend_analysis = (
                            f"Pada tahun {selected_year}, bulan <span style='color: green;'><b>{max_month}</b></span> "
                            f"mengalami donasi tertinggi dengan total donasi sebesar <span style='color: green;'><b>{format_rupiah(max_donation)}</b></span>, "
                            f"sementara bulan <span style='color: red;'><b>{min_month}</b></span> mengalami donasi terendah dengan total donasi sebesar "
                            f"<span style='color: red;'><b>{format_rupiah(min_donation)}</b></span>."
                        )
                        st.markdown(trend_analysis, unsafe_allow_html=True)

                else:
                    st.warning("Masukkan nama donatur untuk melihat tren donasi.")
