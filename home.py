import streamlit as st

def app():
    st.title("Daily & Hourly Flood Monitoring & Forecasting")
    
    st.markdown("<h2>Getting Started</h2>", unsafe_allow_html=True)

    st.markdown("<p>This platform presents tools that can be used to mathematically quantify floods at daily and hourly timescales. It also offers demonstration of Artificial Intelligence models developed using these mathematical methods to forecast floods.</p><p>The tools are based on the rationale that flood on any day is dependent on both the current and antecedent days precipitations with the effect of previous days precipitation on current days flood gradually decaying based on a time-dependent reduction function due to interaction of hydrological conditions such as evapotranspiration, seepage and surface run-off.</p><p>As of 2020, the daily Flood Index has been applied in Australia, South Korea, Iran, Bangladesh and Fiji and has been generally accepted as a suitable tool for flood monitoring while the hourly water resources index has been applied in Australia and South Korea.</p><p>To start computation, navigate to the tool by using the sidebar navigation.", unsafe_allow_html=True)
    st.image("media/flood1.jpg", caption="Source: Pixabay", width=700)
    st.markdown("<h2>Related Research</h2>", unsafe_allow_html=True)

    st.markdown("<ol>" +
        "<li>Moishin, M, Deo, RC, Prasad, R, Raj, N & Abdulla, S 2020, '<a href='https://doi.org/10.1007/s00477-020-01899-6' target='_blank'>Development of Flood Monitoring Index for daily flood risk evaluation: case studies in Fiji</a>', Stochastic Environmental Research and Risk Assessment, pp. 1-16.</li>" +
        "<li>Deo, RC, Adamowski, JF, Begum, K, Salcedo-Sanz, S, Kim, D-W, Dayal, KS & Byun, H-R 2018, '<a href='https://doi.org/10.1007/s00704-018-2657-4'  target='_blank'>Quantifying flood events in Bangladesh with a daily-step flood monitoring index based on the concept of daily effective precipitation</a>', Theoretical and Applied Climatology, vol. 137, no. 1-2, pp. 1201-15.</li>" +
        "<li>Deo, RC, Byun, HR, Kim, GB & Adamowski, JF 2018, '<a href='https://doi.org/10.1007/s10661-018-6806-0'  target='_blank'>A real-time hourly water index for flood risk monitoring: Pilot studies in Brisbane, Australia, and Dobong Observatory, South Korea</a>', Environ Monit Assess, vol. 190, no. 8, p. 450.</li>" +
        "<li>Deo, RC, Byun, H-R, Adamowski, JF & Kim, D-W 2015, '<a href='https://doi.org/10.1007/s11269-015-1046-3'  target='_blank'>A Real-time Flood Monitoring Index Based on Daily Effective Precipitation and its Application to Brisbane and Lockyer Valley Flood Events</a>', Water Resources Management, vol. 29, no. 11, pp. 4075-93.</li>" +
        "<li>Deo, R, Byun, H, Adamowski, J & Kim, D 2014, '<a href='http://www.changes-itn.eu/Conference/Programme/DetailedProgramme/tabid/157/Default.aspx'  target='_blank'>Diagnosis of flood events in Brisbane (Australia) using a flood index based on daily effective precipitation</a>', International Conference: Analysis and Management of Changing Risks for Natural Hazards, European Commission, 7th Framework Programme, Marie Curie Actions …, pp. AP20-1.</li>" +
        "<li>Nosrati, K, Saravi, MM & Shahbazi, A 2010, '<a href='https://link.springer.com/chapter/10.1007/978-3-540-95991-5_127'  target='_blank'>Investigation of Flood Event Possibility over Iran Using Flood Index</a>', in Survival and Sustainability, Springer, pp. 1355-61.</li>" +
        "<li>Byun, H-R & Lee, D-K 2002, '<a href='https://doi.org/10.2151/jmsj.80.33'  target='_blank'>Defining three rainy seasons and the hydrological summer monsoon in Korea using available water resources index</a>', Journal of the Meteorological Society of Japan. Ser. II, vol. 80, no. 1, pp. 33-44.</li>" +
        "<li>Byun, H-R & Chung, J-S 1998, '<a href='https://www.koreascience.or.kr/article/JAKO199811920062762.page'  target='_blank'>Quantified diagnosis of flood possibility by using effective precipitation index</a>', Journal of Korea Water Resources Association, vol. 31, no. 6, pp. 657-65.</li>" +
        "</ol>", unsafe_allow_html=True)

    st.markdown("<h3>Disclaimer</h3>", unsafe_allow_html=True)
    st.markdown("<p>The developers and researchers of this web application do not take any responsibility for any results produced by the tools presented in the web application and are not liable for any damages caused. By using the tools, you are agreeing to have read this disclaimer.</p>", unsafe_allow_html=True)