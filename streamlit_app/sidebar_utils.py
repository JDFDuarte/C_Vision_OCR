# Get the current date
from datetime import datetime
import calendar
import streamlit as st

def display_calendar_in_sidebar():
    # Get the current date
    current_date = datetime.now()

    # Get the current month, year, and day
    current_year = current_date.year
    current_month = current_date.month
    current_day = current_date.day

    # Create a custom calendar formatter
    class CustomHTMLCalendar(calendar.HTMLCalendar):
        def formatday(self, day, weekday):
            if day == 0:
                return '<td class="noday">&nbsp;</td>'
            elif day == current_day:
                return f'<td class="today"><strong>{day}</strong></td>'
            return f'<td>{day}</td>'

    # Create a custom HTML calendar for the current month
    cal = CustomHTMLCalendar(calendar.MONDAY)
    month_calendar = cal.formatmonth(current_year, current_month)

    # Add calendar to the sidebar
    st.sidebar.title("Calendar")
    st.sidebar.markdown(month_calendar, unsafe_allow_html=True)

    # Add custom CSS to style the calendar
    st.markdown("""
    <style>
    table.month {
        border-collapse: collapse;
        margin: 0 auto;
    }
    table.month th, table.month td {
        border: 1px solid #ddd;
        padding: 5px;
        text-align: center;
    }
    table.month td.today {
        background-color: "rgba(232, 234, 237, 0.4)";
        border-radius: 50%;
    }
    </style>
    """, unsafe_allow_html=True)