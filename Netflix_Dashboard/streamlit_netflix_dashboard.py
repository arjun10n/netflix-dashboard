import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title=" Netflix Data Dashboard", layout="wide")

# ===============================
# DATA LOADING + CLEANING
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Arjun\OneDrive\Desktop\Netflix_Dashboard\netflix_titles.csv.csv")

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Fill missing text columns
    text_columns = ['director', 'cast', 'country', 'rating']
    for col in text_columns:
        if col in df.columns:
            df[col].fillna('Unknown', inplace=True)

    # Fill missing release year
    if 'release_year' in df.columns:
        df['release_year'].fillna(df['release_year'].median(), inplace=True)

    # Extract decade
    df['decade'] = (df['release_year'] // 10) * 10

    # Convert duration to numeric
    def convert_duration(x):
        if 'Season' in str(x):
            return int(x.split()[0]) * 60
        elif 'min' in str(x):
            return int(x.split()[0])
        else:
            return 0

    df['duration_min'] = df['duration'].apply(convert_duration)
    return df


df = load_data()

# ===============================
# SIDEBAR FILTERS
# ===============================
st.sidebar.header(" Filter Options")

countries = ["All"] + sorted(df['country'].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Select Country", countries)

min_year, max_year = int(df['release_year'].min()), int(df['release_year'].max())
selected_year = st.sidebar.slider("Select Release Year Range", min_year, max_year, (2010, max_year))

# Apply filters
filtered_df = df.copy()
if selected_country != "All":
    filtered_df = filtered_df[filtered_df['country'] == selected_country]

filtered_df = filtered_df[
    (filtered_df['release_year'] >= selected_year[0]) &
    (filtered_df['release_year'] <= selected_year[1])
]

# ===============================
# HEADER
# ===============================
st.title("🎬 Netflix Global Insights Dashboard")
st.markdown(
    f"##### An interactive data exploration of Netflix titles "
    f"filtered by **{selected_country}** ({selected_year[0]}–{selected_year[1]})"
)

# ===============================
# TOP SUMMARY STATS
# ===============================
col1, col2, col3 = st.columns(3)
col1.metric("📺 Total Titles", len(filtered_df))
col2.metric("🎞️ Movies", len(filtered_df[filtered_df['type'] == 'Movie']))
col3.metric("📺 TV Shows", len(filtered_df[filtered_df['type'] == 'TV Show']))

# ===============================
# TAB SECTIONS
# ===============================
tab1, tab2, tab3 = st.tabs([" Exploratory Data Analysis", " Machine Learning", " Summary"])

# =========================================
# TAB 1: EDA
# =========================================
with tab1:
    st.subheader("Visual Insights")

    col1, col2 = st.columns(2)

    # Top countries by titles
    if selected_country == "All":
        with col1:
            top_countries = df['country'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=top_countries.values, y=top_countries.index, palette='mako', ax=ax)
            ax.set_title(" Top 10 Countries by Total Titles")
            st.pyplot(fig)
    else:
        with col1:
            type_counts = filtered_df['type'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=type_counts.index, y=type_counts.values, palette='Set2', ax=ax)
            ax.set_title(f" Movies vs TV Shows in {selected_country}")
            st.pyplot(fig)

    # Trend by release year
    with col2:
        trend = filtered_df.groupby('release_year').size()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(x=trend.index, y=trend.values, marker='o', ax=ax)
        ax.set_title("Trend of Netflix Titles Over Time")
        ax.set_xlabel("Release Year")
        ax.set_ylabel("Number of Titles")
        st.pyplot(fig)

    # Duration comparison
    st.markdown("### ⏱ Duration Comparison between Movies & TV Shows")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x='type', y='duration_min', data=filtered_df, palette='coolwarm', ax=ax)
    st.pyplot(fig)

# =========================================
# TAB 2: MACHINE LEARNING
# =========================================
with tab2:
    st.subheader("Machine Learning Models: Predicting Type")

    df_model = df.copy()
    le = LabelEncoder()
    df_model['country_encoded'] = le.fit_transform(df_model['country'].astype(str))
    df_model['rating_encoded'] = le.fit_transform(df_model['rating'].astype(str))

    X = df_model[['release_year', 'duration_min', 'country_encoded', 'rating_encoded']]
    y = le.fit_transform(df_model['type'].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    log_acc = accuracy_score(y_test, y_pred_log)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)

    st.write(f"**Logistic Regression Accuracy:** {log_acc:.2f}")
    st.write(f"**Random Forest Accuracy:** {rf_acc:.2f}")

    acc_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [log_acc, rf_acc]
    })

    fig, ax = plt.subplots()
    sns.barplot(data=acc_df, x="Model", y="Accuracy", palette="crest", ax=ax)
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# =========================================
# TAB 3: SUMMARY
# =========================================
with tab3:
    st.subheader("Key Findings Summary")
    st.markdown("""
    - Netflix’s **content production** peaked around **2015–2020**.  
    - The **United States and India** are top contributors globally.  
    - Movies generally have **shorter durations** than TV Shows.  
    - The **Logistic Regression model** achieved high performance for classifying content type,  
      but **Random Forest** performed slightly better.  
    - There’s a visible global **shift toward international diversity** in content.  
    """)

st.success("✅ Dashboard and ML Models Loaded Successfully!")
