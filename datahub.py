import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import json
import requests
from streamlit_lottie import st_lottie

st.set_page_config(
    page_icon=":book:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def data():
    
    import streamlit as st
    
        
    st.title("DATAHUB")
    def load_lottieurl(url: str):
            import requests
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()
    with st.sidebar:
        selected = option_menu(
            menu_title = None,
            options = ["homepage","createmodel","datavisual","mail","stockdashboard","tableau","Own Visual"],
            icons = ["book","back","clipboard-data-fill","envelope-at-fill","body-text","bar-chart"],
            menu_icon="cast",
        )
    
    if selected ==  "homepage" :
        import streamlit as st
        import json
        import requests
        from streamlit_lottie import st_lottie

        def load_lottieurl(url: str):
                r = requests.get(url)
                if r.status_code != 200:
                    return None
                return r.json()

        lottie_computer = load_lottieurl("https://lottie.host/da83f0ff-a69b-40c4-8691-8297ac96bdf1/j39KODV24b.json")

        st_lottie(lottie_computer)


        st.subheader("AutoStreamML - Your Automated Machine Learning Solution")
        st.write("""
            Welcome to AutoStreamML, your one-stop solution for automating the machine learning pipeline. This user-friendly application is powered by Streamlit, Pandas Profiling, and Pycaret, offering a seamless and efficient way to analyze data, build models, and save them for future use.""")

        st.subheader("Key Features:")

        st.subheader("Data Upload:") 
        st.write("""Begin your machine learning journey by selecting "Upload." You can easily upload your dataset, making it ready for analysis and modeling.""")

        st.subheader("Automated EDA (Exploratory Data Analysis):") 
        st.write("""Choose "Profiling" to generate automated exploratory data analysis reports. Gain valuable insights into your dataset quickly and effortlessly.""")

        st.subheader("Machine Learning Made Simple:") 
        st.write("""Dive into the world of machine learning with the "Modelling" option. Select your target variable, train models, and compare their performance with a single click. AutoStreamML streamlines the entire process, making machine learning accessible to all.""")

        st.subheader("Effortless Model Download:") 
        st.write("""Once you've trained your models, use the "Download" option to save the best-performing model as "trained_model.pkl." This model is ready for deployment and further analysis.""")


        lottie_visual = load_lottieurl("https://lottie.host/f1568a9a-f929-4842-8b6c-3218546acff5/tSw6H4MsWe.json")
        st_lottie(lottie_visual)

        st.subheader("Excel Plotter - Visualize Your Data with Ease")
        st.write("""
            Introducing the Excel Plotter, a versatile data visualization tool that empowers you to explore and visualize your Excel data effortlessly. This Streamlit-powered application simplifies the process of creating insightful charts and graphs from your Excel files.""")

        st.subheader("Key Features:")

        st.subheader("Data Grouping:") 
        st.write(""" If you have multiple columns, you can group your data by selecting the columns of interest. This feature simplifies the process of analyzing grouped data.""")

        st.subheader("Interactive Charts:") 
        st.write("""The charts are interactive and responsive. You can zoom in, pan, and hover over data points to explore your information in detail.""")

        st.subheader("Download Your Visualizations: ") 
        st.write("""Excel Plotter offers easy download options. You can download your visualizations as Excel files or HTML documents for sharing and further analysis.""")


        lottie_tableau = load_lottieurl("https://lottie.host/4be4bc84-61cc-4dbd-9640-5c8a7a98727f/jNeAIyVpY7.json")

        st_lottie(lottie_tableau)

        st.subheader("PyGWalker - Your Data Visualization Wizard")
        st.write("""
            Welcome to PyGWalker, your trusted companion for effortlessly visualizing data in a way that suits your needs. This Streamlit-powered application is designed to simplify the process of creating stunning data visualizations from your XLSX or CSV files.""")

        st.subheader("Key Features:")

        st.subheader("File Selection:") 
        st.write(""" Get started by selecting the data file you want to visualize. PyGWalker supports both XLSX and CSV file formats, ensuring flexibility with your data sources.""")

        st.subheader("Visualize Your Data:") 
        st.write(""" Once your file is uploaded, PyGWalker instantly loads your data and prepares it for visualization. You can now explore and understand your data visually.""")

        st.subheader("Chart Selection: ") 
        st.write("""PyGWalker offers a selection of charts to choose from. Whether you're looking for bar charts, line plots, scatter plots, or other chart types, you have the flexibility to pick the one that best represents your data.""")

        st.subheader("Customization: ") 
        st.write("""Customize your charts further by adjusting settings and configurations to meet your specific requirements. PyGWalker puts you in control of how your data is presented.""")


        lottie_stock = load_lottieurl("https://lottie.host/5f280b2b-be0c-4cca-a535-c0bbe3bdb0e9/AixqbYSEgX.json")

        st_lottie(lottie_stock)

        st.subheader("Stock Dashboard - Visualize Stock Data with Ease")
        st.write("""The Stock Dashboard is your all-in-one tool for gaining insights into stock market data effortlessly. Powered by Streamlit and various data sources, this application simplifies the process of tracking stock prices, analyzing performance, and staying updated with the latest news.""")

        st.subheader("Key Features:")

        st.subheader("Customized Stock Analysis:") 
        st.write(""" Begin by entering the stock ticker of your choice. The Stock Dashboard allows you to specify the start and end dates for your analysis, providing flexibility in data selection.""")

        st.subheader("Interactive Price Chart: ") 
        st.write(""" The application fetches historical stock price data and displays it using an interactive line chart. You can zoom in, pan, and hover over data points to examine stock performance over time.""")
        st.subheader("Multi-tab Interface:  ") 
        st.write("""The Stock Dashboard offers three tabs for a comprehensive analysis:\n
    Pricing Data:\n
    Dive into stock pricing details, including daily closing prices and percentage changes. Analyze annual returns, standard deviation, and risk-adjusted returns.\n
    Fundamental Data:\n
    Explore fundamental financial data, including balance sheets, income statements, and cash flow statements. Gain insights into the company's financial health.\n
    Top 10 News:\n
    Stay informed with the latest news related to the selected stock. The news section provides a summary, sentiment analysis, and publication details.""")


        lottie_mail = load_lottieurl("https://lottie.host/a5dc2b7b-f432-4f9b-bd32-33ef7db199b0/78ElTsPjqe.json")

        st_lottie(lottie_mail)

        st.subheader("Email Sender Web Application")
        st.write("""The "Email Sender Web Application" simplifies the process of sending emails via Gmail, offering these four key features:""")

        st.subheader("Key Features:")

        st.subheader("User-Friendly Interface:") 
        st.write("""The application boasts an intuitive and easy-to-use interface, making it accessible to users of all technical levels.""")

        st.subheader("Secure Login:") 
        st.write(""" Users provide their Gmail email address and securely input their password (masked) for authentication.""")

        st.subheader("Compose and Send Emails:") 
        st.write("""Users can compose emails by specifying the recipient's email address, subject, and email content in the provided fields.""")

        st.subheader(" Error Handling and Audio Feedback: ") 
        st.write("""The application includes robust error handling:
    It prompts users to complete all required fields if left empty.\n
    It checks for an internet connection and informs users to connect if needed.\n
    In case of incorrect Gmail credentials, it displays an error message.\n
    It provides audio feedback using text-to-speech technology, announcing "Email sent successfully" when an email is sent without errors and vocalizing error messages for user understanding.""")
        
        

                

    if selected == "createmodel" :
            import plotly.express as px
            import streamlit as st
            import pandas as pd
            import numpy as np
            import os
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression, LogisticRegression
            from sklearn.cluster import KMeans
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import mean_squared_error, accuracy_score
            import joblib

            if 'sourcedata.csv' in os.listdir():
                df = pd.read_csv("sourcedata.csv", index_col=None)
            else:
                df = pd.DataFrame() 

            st.sidebar.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
            st.sidebar.title("AutoStreamML")
            choice = st.sidebar.radio("Navigation", ["Upload", "Modelling (Linear Regression)",
                                                    "Modelling (Logistic Regression)", "Modelling (KMeans Classification)",
                                                    "Modelling (Decision Tree)"])
            st.sidebar.info("This application allows you to build an automated ML pipeline using Streamlit, Pandas Profiling, and PyCaret.")

            if choice == "Upload":
                st.title("Upload Your Data for Modelling")
                file = st.file_uploader("Upload Your Dataset HERE")
                if file:
                    df = pd.read_csv(file)
                    df.to_csv("sourcedata.csv", index=False)
                    st.dataframe(df)

            if choice.startswith("Modelling"):
                st.title("Machine Learning")
                st.markdown("<hr>", unsafe_allow_html=True)

                # Data Cleaning option
                if choice.endswith("(Linear Regression)") or choice.endswith("(Logistic Regression)") or \
                        choice.endswith("(KMeans Classification)") or choice.endswith("(Decision Tree)"):
                    selected_option = st.radio("Select an option:", ["DATA CLEANING", "NO DATA CLEANING"])

                    if selected_option == "DATA CLEANING":
                        st.subheader("FULL DATA FRAME")
                        st.dataframe(df)
                    if not df.empty:
                        target = st.selectbox("Select Your Target", df.columns)
                        features = df.drop(columns=[target])

                        columns_to_drop = st.multiselect("Select columns to drop", df.columns)
                        new_df = df.drop(columns=columns_to_drop)

                        X = new_df.drop(columns=[target])
                        y = new_df[target]

                        if choice.endswith("(Linear Regression)"):
                            model = LinearRegression()
                        elif choice.endswith("(Logistic Regression)"):
                            model = LogisticRegression()
                        elif choice.endswith("(KMeans Classification)"):
                            model = KMeans(n_clusters=2, random_state=42)
                        elif choice.endswith("(Decision Tree)"):
                            model = DecisionTreeClassifier()

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        st.write("Predictions:")
                        st.write(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))

                        if 'Classification' in choice:
                            accuracy = accuracy_score(y_test, y_pred)
                            st.write("Accuracy:", accuracy)

                        if st.button("Download Model"):
                            model_filename = f"{choice.lower().replace(' ', '_')}_model.joblib"
                            joblib.dump(model, model_filename)
                            st.success(f"Model downloaded as {model_filename}")

                            
                        # Scatter plots for features vs target
                        st.subheader("Bar Plots for Features by Target Class")
                        for feature in new_df.columns:
                            if feature != target:
                                fig = px.histogram(new_df, x=feature, color=target, barmode='group', title=f"Distribution of {feature} by {target}")
                                st.plotly_chart(fig)




                    else:
                        st.warning("Please upload a dataset first.")

                elif selected_option == "NO DATA CLEANING":
                    if not df.empty:
                        target = st.selectbox("Select Your Target", df.columns)
                        y = df[target]
                        X = df.drop(columns=[target])

                        if choice.endswith("(Linear Regression)"):
                            model = LinearRegression()
                        elif choice.endswith("(Logistic Regression)"):
                            model = LogisticRegression()
                        elif choice.endswith("(KMeans Classification)"):
                            model = KMeans(n_clusters=2, random_state=42)
                        elif choice.endswith("(Decision Tree)"):
                            model = DecisionTreeClassifier()

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        st.write("Predictions:")
                        st.write(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))

                        if 'Classification' in choice:
                            accuracy = accuracy_score(y_test, y_pred)
                            st.write("Accuracy:", accuracy)

                        if st.button("Download Model"):
                            model_filename = f"{choice.lower().replace(' ', '_')}_model.joblib"
                            joblib.dump(model, model_filename)
                            st.success(f"Model downloaded as {model_filename}")

                        st.subheader("Scatter Plots for Features vs Target")
                        for feature in X.columns:
                            fig = px.scatter(df, x=X[feature], y=target, color=target, title=f"{feature} vs {target}", trendline="ols")
                            st.plotly_chart(fig)
                    else:
                        st.warning("Please upload a dataset first.")


                            

                    

            



    if selected == "datavisual" :
        import streamlit as st 
        import pandas as pd 
        import plotly.express as px 
        import base64  
        from io import StringIO, BytesIO  
        import seaborn as sns
        from streamlit_lottie import st_lottie

        visual = load_lottieurl("https://lottie.host/e2ad82b5-c5eb-48d6-9e83-59ed32d991d6/uhgT0qz7DM.json")

        st_lottie(visual)

        def generate_excel_download_link(df):
            # Credit Excel: https://discuss.streamlit.io/t/how-to-add-a-download-excel-csv-function-to-a-button/4474/5
            towrite = BytesIO()
            df.to_excel(towrite, encoding="utf-8", index=False, header=True) 
            towrite.seek(0)  # reset pointer
            b64 = base64.b64encode(towrite.read()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="data_download.xlsx">Download Excel File</a>'
            return st.markdown(href, unsafe_allow_html=True)

        def generate_html_download_link(fig):
            # Credit Plotly: https://discuss.streamlit.io/t/download-plotly-plot-as-html/4426/2
            towrite = StringIO()
            fig.write_html(towrite, include_plotlyjs="cdn")
            towrite = BytesIO(towrite.getvalue().encode())
            b64 = base64.b64encode(towrite.read()).decode()
            href = f'<a href="data:text/html;charset=utf-8;base64, {b64}" download="plot.html">Download Plot</a>'
            return st.markdown(href, unsafe_allow_html=True)


        
        st.title('Excel Plotter 📈')
        st.subheader('Feed me with your Excel file')

        uploaded_file = st.file_uploader('Choose a file', type=['xlsx', 'csv'])
        if uploaded_file:
            st.markdown('---')
            if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 
                try:
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                except Exception as e:
                    st.error(f"Error reading XLSX file: {e}")
            elif uploaded_file.type == 'text/csv': 
                df = pd.read_csv(uploaded_file)
            st.dataframe(df)

            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

            for selected_column in numerical_columns:
                st.subheader(f"Chart for {selected_column}")
                unique_key = f"{selected_column}_chart"
                choice = st.selectbox(f"SELECT CHART YOU WANT ({selected_column})", ["FUNNEL", "SCATTER", "PIE", "BAR", "LINE", "BOX", "VIOLIN", "HISTOGRAM", "AREA"], key=unique_key)
                if choice is not None:
                    # -- PLOT CHART
                    if choice == "FUNNEL":
                        fig = px.funnel(df, x=selected_column)
                    elif choice == "SCATTER":
                        fig = px.scatter(
                            df,
                            x=df.index,
                            y=selected_column,
                            template='plotly_white',
                            title=f'<b>{selected_column} by Index</b>'
                        )
                    elif choice == "PIE":
                        fig = px.pie(
                            df,
                            values=selected_column,
                            template='plotly_white',
                            title=f'<b>{selected_column} Distribution</b>'
                        )
                    elif choice == "BAR":
                        fig = px.bar(
                            df,
                            x=df.index,
                            y=selected_column,
                            template='plotly_white',
                            title=f'<b>{selected_column} by Index</b>'
                        )
                    elif choice == "LINE":
                        fig = px.line(
                            df,
                            x=df.index,
                            y=selected_column,
                            template='plotly_white',
                            title=f'<b>{selected_column} Line Chart</b>'
                        )
                    elif choice == "BOX":
                        fig = px.box(
                            df,
                            y=selected_column,
                            template='plotly_white',
                            title=f'<b>{selected_column} Box Plot</b>'
                        )
                    elif choice == "VIOLIN":
                        fig = px.violin(
                            df,
                            y=selected_column,
                            template='plotly_white',
                            title=f'<b>{selected_column} Violin Plot</b>'
                        )
                    elif choice == "HISTOGRAM":
                        fig = px.histogram(
                            df,
                            x=selected_column,
                            template='plotly_white',
                            title=f'<b>{selected_column} Histogram</b>'
                        )
                    elif choice == "AREA":
                        fig = px.area(
                            df,
                            x=df.index,
                            y=selected_column,
                            template='plotly_white',
                            title=f'<b>{selected_column} Area Chart</b>'
                        )

                    st.plotly_chart(fig)
                    
                    st.subheader(f'Downloads for {selected_column}:')
                    generate_html_download_link(fig)
                    st.info("Task completed!!!!")
                else:
                    st.error("Could you please select the chart type")

            
            if st.button("Generate Pair Plot"):
                st.subheader("Pair Plot for All Numeric Columns")
                pairplot_fig = sns.pairplot(df)
                st.pyplot(pairplot_fig)
                #generate_html_download_link(pairplot_fig)




    if selected == "tableau" :
        import streamlit as st
        import pandas as pd
        import pygwalker as pyg
        from streamlit_lottie import st_lottie
        import streamlit.components.v1 as components

        tableau = load_lottieurl("https://lottie.host/3591b49e-3ae2-4137-945a-d741f9064a49/c8HNZeYMZB.json")

        st_lottie(tableau)
        

        st.title('Tableau')
        choice = st.selectbox("Select chart type:", ["XLSX", "CSV"])

        if choice == "XLSX":
            uploaded_file = st.file_uploader('Choose an XLSX file', type='xlsx')
            if uploaded_file is None:
                st.error("Please select a dataset.")
            else:
                df = pd.read_excel(uploaded_file)
                pyg_output = pyg.walk(df, env="streamlit", dark='dark')
                pyg_html = pyg_output.to_html()  
                components.html(pyg_html, height=1000, scrolling=True)

        if choice == "CSV":
            uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
            if uploaded_file is None:
                st.error("Please select a dataset.")
            else:
                df = pd.read_csv(uploaded_file)
                pyg_output = pyg.walk(df, env="streamlit", dark='dark')
                pyg_html = pyg_output.to_html()  
                components.html(pyg_html, height=1000, scrolling=True)


    if selected == "stockdashboard" : 
        import plotly.express as px
        import base64
        from io import BytesIO
        import streamlit as st
        import pandas as pd
        import numpy as np
        import yfinance as yf
        from streamlit_lottie import st_lottie
        from alpha_vantage.fundamentaldata import FundamentalData
        from stocknews import StockNews

        def generate_excel_download_link(df):
            # Credit Excel: https://discuss.streamlit.io/t/how-to-add-a-download-excel-csv-function-to-a-button/4474/5
            towrite = BytesIO()
            df.to_excel(towrite, index=False, header=True)
            towrite.seek(0)  # reset pointer
            b64 = base64.b64encode(towrite.read()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="data_download.xlsx">Download Excel File</a>'
            return st.markdown(href, unsafe_allow_html=True)

        try:
            
            stock = load_lottieurl("https://lottie.host/012d3055-a168-463a-8bbf-7236e6319768/UTNFd4Xl1P.json")
            st_lottie(stock)
            
            st.title("Stock Dashboard")
            ticker = st.sidebar.text_input("Ticker")
            start_date = st.sidebar.date_input("Start Date")
            end_date = st.sidebar.date_input("End Date")

            data = yf.download(ticker,start = start_date, end = end_date)
            fig = px.line(data, x = data.index, y = data["Adj Close"], title = ticker)
            st.plotly_chart(fig)

            pricing_data, fundamental_data, news = st.tabs(["Pricing Data","Fundamental Data", "Top 10 News"])
            
            with pricing_data:
                st.write("Price")
                data2 = data
                data2["% Change"] = data["Adj Close"]/data["Adj Close"].shift(1) - 1
                data2.dropna(inplace = True)
                st.write(data)
                annual_return = data2["% Change"].mean()*252*100
                st.write("Annual Return is ",annual_return,"%")
                stdev = np.std(data2["% Change"])*np.sqrt(252)
                st.write("Standard Deviation is ", stdev*100,"%")
                st.write("Risk Adj, Return is ",annual_return/(stdev*100))
                generate_excel_download_link(data)

            with fundamental_data:
                key = "9Z1Z01K4E1DNN1AI"
                fd = FundamentalData(key, output_format = "pandas")
                st.subheader("Balance Sheet")
                balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
                bs = balance_sheet.T[2:]
                bs.columns = list(balance_sheet.T.iloc[0])
                st.write(bs)
                generate_excel_download_link(bs)
                st.subheader("Income Statement")
                income_statement = fd.get_income_statement_annual(ticker)[0]
                is1 = income_statement.T[2:]
                is1.columns = list(income_statement.T.iloc[0])
                st.write(is1)
                generate_excel_download_link(is1)
                st.subheader("Cash Flow Statement")
                cash_flow = fd.get_cash_flow_annual(ticker)[0]
                cf = cash_flow.T[2:]
                cf.columns = list(cash_flow.T.iloc[0])
                st.write(cf)
                generate_excel_download_link(cf)

            with news:
                st.header(f"News of {ticker}")
                sn = StockNews(ticker, save_news = False)
                df_news = sn.read_rss()
                for i in range(10):
                    st.write(df_news["published"][i])
                    st.write(df_news["title"][i])
                    st.write(df_news["summary"][i])
                    title_sentiment = df_news["sentiment_title"][i]
                    st.write(f"Title Sentiment {title_sentiment}")
                    news_sentiment = df_news["sentiment_summary"][i]
                    st.write(f"News Sentiment {news_sentiment}")

        except Exception as e:
            st.error("No  data available for this ticker ")

    if selected == "mail":
        import streamlit as st
        from io import BytesIO
        import pandas as pd
        import yagmail
        import ssl

        ssl_context = ssl.create_default_context()

        yag = None

        sender_email = st.text_input("Enter your email:")
        password = st.text_input("Enter your password:", type="password")
        receiver_email = st.text_input("Enter receiver's email:")
        sub = st.text_input("Enter the Subject:")
        bod = st.text_input("Enter the Body:")

        uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

        if sender_email and password and receiver_email and uploaded_file:
            df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)
            
            excel_buffer = BytesIO()
            if uploaded_file.name.endswith("xlsx"):
                df.to_excel(excel_buffer, index=False)
            else:
                df.to_csv(excel_buffer, index=False)
            excel_buffer.seek(0)

            yag = yagmail.SMTP(sender_email, password)

        if yag is not None and st.button("S E N D"):
            subject = sub
            body = bod
            
            print("Attachments:", uploaded_file)
            
            yag.send(
                to=receiver_email,
                subject=subject,
                contents=body,
                attachments=uploaded_file
            )

            
            yag.close()
            st.success("Email sent successfully!")
    

    if selected ==  "Own Visual":
        st.title('Tableau')
        choice = st.selectbox("Select File type:", ["XLSX", "CSV"])

        if choice == "XLSX":
            uploaded_file = st.file_uploader('Choose an XLSX file', type='xlsx')
            if uploaded_file is None:
                st.error("Please select a dataset.")
            else:
                df = pd.read_excel(uploaded_file)

        if choice == "CSV":
            uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
            if uploaded_file is None:
                st.error("Please select a dataset.")
            else:
                df = pd.read_csv(uploaded_file)




data()


