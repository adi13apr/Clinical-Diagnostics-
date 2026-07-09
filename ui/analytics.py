import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def show_analytics(data_service, user_email):
    """Display analytics dashboard with charts and statistics"""
    
    st.title("📊 Analytics Dashboard")
    
    # Fetch user's diagnostic history
    history = data_service.get_history(user_email)
    
    if not history.data or len(history.data) == 0:
        st.info("No diagnostic data available yet. Start analyzing X-rays to see analytics.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(history.data)
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Key Metrics Row
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Diagnoses", len(df))
    
    with col2:
        avg_confidence = df['confidence_score'].mean() if 'confidence_score' in df.columns else 0
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    with col3:
        normal_count = len(df[df['prediction'] == 'Normal'])
        st.metric("Normal Cases", normal_count)
    
    with col4:
        abnormal_count = len(df[df['prediction'] != 'Normal'])
        st.metric("Abnormal Cases", abnormal_count)
    
    # Charts Row 1
    st.divider()
    chart_col1, chart_col2 = st.columns(2)
    
    # Diagnosis Distribution Pie Chart
    with chart_col1:
        st.subheader("Diagnosis Distribution")
        diagnosis_counts = df['prediction'].value_counts()
        
        colors_map = {
            'Normal': '#00D084',
            'Pneumonia': '#FFA500',
            'Tuberculosis': '#FF0000'
        }
        colors = [colors_map.get(diagnosis, '#808080') for diagnosis in diagnosis_counts.index]
        
        fig_pie = px.pie(
            values=diagnosis_counts.values,
            names=diagnosis_counts.index,
            color=diagnosis_counts.index,
            color_discrete_map=colors_map,
            title="Case Distribution by Diagnosis"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Confidence Score Distribution
    with chart_col2:
        st.subheader("Confidence Scores")
        if 'confidence_score' in df.columns:
            df['confidence_score_clean'] = pd.to_numeric(df['confidence_score'], errors='coerce')
            fig_hist = px.histogram(
                df,
                x='confidence_score_clean',
                nbins=20,
                title="Confidence Score Distribution",
                labels={'confidence_score_clean': 'Confidence (%)'},
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Confidence data not available")
    
    # Charts Row 2
    chart_col3, chart_col4 = st.columns(2)
    
    # Diagnoses Over Time
    with chart_col3:
        st.subheader("Diagnoses Over Time")
        df_time = df.groupby([df['created_at'].dt.date, 'prediction']).size().reset_index(name='count')
        fig_line = px.bar(
            df_time,
            x='created_at',
            y='count',
            color='prediction',
            color_discrete_map=colors_map,
            title="Daily Diagnosis Trend",
            labels={'created_at': 'Date', 'count': 'Number of Cases'}
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Severity Breakdown
    with chart_col4:
        st.subheader("Severity Breakdown")
        severity_map = {
            'Normal': 'Normal',
            'Pneumonia': 'Pneumonia',
            'Tuberculosis': 'Tuberculosis'
        }
        df['severity'] = df['prediction'].map(severity_map)
        severity_counts = df['severity'].value_counts()
        
        fig_bar = px.bar(
            x=severity_counts.index,
            y=severity_counts.values,
            color=severity_counts.index,
            color_discrete_map=colors_map,
            title="Cases by Severity Level",
            labels={'x': 'Severity', 'y': 'Count'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Detailed Statistics Table
    st.divider()
    st.subheader("Detailed Statistics by Diagnosis")
    
    stats_data = []
    for diagnosis in df['prediction'].unique():
        diagnosis_df = df[df['prediction'] == diagnosis]
        avg_conf = diagnosis_df['confidence_score'].mean() if 'confidence_score' in diagnosis_df.columns else 0
        stats_data.append({
            'Diagnosis': diagnosis,
            'Count': len(diagnosis_df),
            'Percentage': f"{(len(diagnosis_df) / len(df) * 100):.1f}%",
            'Avg Confidence': f"{avg_conf:.1f}%",
            'Latest': diagnosis_df['created_at'].max().strftime('%Y-%m-%d %H:%M')
        })
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Recent Activity
    st.divider()
    st.subheader("Recent Diagnoses")
    recent_df = df.nlargest(5, 'created_at')[['patient_name', 'prediction', 'confidence_score', 'created_at']].copy()
    recent_df['created_at'] = recent_df['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
    recent_df.columns = ['Patient ID', 'Diagnosis', 'Confidence (%)', 'Timestamp']
    st.dataframe(recent_df, use_container_width=True, hide_index=True)
