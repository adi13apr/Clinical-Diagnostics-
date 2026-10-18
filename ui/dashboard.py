import streamlit as st
from PIL import Image
from services.pdf_handler import PDFReportGenerator
from services.calibration import REFER_THRESHOLD, REFER_MESSAGE, STATUS_REFER
from datetime import datetime
import pandas as pd
import time

def get_severity_badge(diagnosis):
    """Return severity level and color based on diagnosis"""
    severity_map = {
        'Normal': {'level': '✅ NORMAL', 'color': '#00D084'},  # Green
        'Pneumonia': {'level': '⚠️ PNEUMONIA', 'color': '#FFA500'},  # Orange
        'Tuberculosis': {'level': '🔴 TUBERCULOSIS', 'color': '#FF0000'}  # Red
    }
    return severity_map.get(diagnosis, {'level': diagnosis, 'color': '#808080'})

def get_status_label(status):
    """Short triage label for tables"""
    return '⚠️ Refer' if status == STATUS_REFER else '✅ Confident'

def show_status(result):
    """Banner telling the clinician whether the prediction can be relied on"""
    confidence = result['confidence'] * 100
    if result['status'] == STATUS_REFER:
        st.warning(f"**{REFER_MESSAGE.capitalize()}.** Calibrated confidence {confidence:.1f}% "
                   f"is below the {REFER_THRESHOLD:.0%} threshold.")
    else:
        st.success(f"**Confident.** Calibrated confidence {confidence:.1f}% "
                   f"meets the {REFER_THRESHOLD:.0%} threshold.")

def show_report(report_result):
    """Render an agent ReportResult"""
    report = report_result.report
    st.caption(report_result.source_line())
    if report_result.cached:
        st.caption("Reused the report already written for this image (no new Gemini requests).")
    if report_result.reason:
        st.caption(f"AI writer not used: {report_result.reason} Showing the template report.")
    st.markdown(f"**Summary.** {report.summary}")
    if report.status == STATUS_REFER:
        st.warning(f"**Recommendation.** {report.recommendation}")
    else:
        st.info(f"**Recommendation.** {report.recommendation}")
    st.markdown("**Limitations**")
    for limitation in report.limitations:
        st.markdown(f"- {limitation}")
    st.caption(report.disclaimer)

def show_single_result(user_email, model_service, image):
    """Prediction, heatmap, AI report and PDF for the last single-image run (kept in session_state)"""
    result = st.session_state['last_result']
    patient_name = st.session_state['last_patient_name']
    diagnosis = result['diagnosis']
    confidence = result['confidence'] * 100
    severity = get_severity_badge(diagnosis)

    # Show diagnosis with color badge and confidence
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Diagnosis", severity['level'])
    with col2:
        st.metric("Confidence (calibrated)", f"{confidence:.1f}%")
    with col3:
        # Show severity indicator
        st.markdown(f"<div style='background-color:{severity['color']};padding:20px;border-radius:10px;text-align:center;color:white;font-weight:bold;'>SEVERITY</div>", unsafe_allow_html=True)

    show_status(result)

    # Show all probabilities
    st.subheader("Prediction Probabilities:")
    for class_name, prob in result['probabilities'].items():
        st.progress(prob, text=f"{class_name}: {prob*100:.1f}%")

    # Heatmap generated when the diagnosis ran
    if st.session_state.get('last_heatmap') is not None:
        st.divider()
        st.subheader("🔥 AI Interpretation Heatmap")
        st.info("Red areas indicate regions that most influenced the AI diagnosis. Darker blue indicates lower influence.")
        st.image(st.session_state['last_heatmap'], caption="Grad-CAM Attention Map - Shows which regions influenced the diagnosis", use_container_width=True)

    # AI report: the LLM explains the classifier's output but can't change it
    st.divider()
    st.subheader("📝 AI Report")
    if st.button("Generate report"):
        from agent.agent import build_llm_client, generate_report_for_image
        with st.spinner("Writing report..."):
            st.session_state['last_report'] = generate_report_for_image(
                model_service, image, prediction=result, llm=build_llm_client())
    report_result = st.session_state.get('last_report')
    if report_result is not None:
        show_report(report_result)

    # Generate and offer PDF download
    st.divider()
    pdf_generator = PDFReportGenerator()
    pdf_data = pdf_generator.generate_diagnosis_report(
        patient_name=patient_name,
        diagnosis=diagnosis,
        confidence=confidence,
        probabilities=result['probabilities'],
        radiologist_email=user_email,
        timestamp=st.session_state['last_timestamp'],
        status=result['status'],
        report=report_result.report if report_result else None,
        report_source=report_result.source_line() if report_result else None
    )

    st.download_button(
        label="📥 Download Report (PDF)",
        data=pdf_data,
        file_name=f"Diagnosis_{patient_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf"
    )

def run_prediction(model_service, prediction_service, image):
    """Run the model, timing it, and log the prediction (logging failures only warn)"""
    start = time.perf_counter()
    result = model_service.predict(image)
    inference_ms = round((time.perf_counter() - start) * 1000)
    
    try:
        prediction_service.log_prediction(result, inference_ms)
    except Exception as e:
        st.warning(f"Could not log prediction: {str(e)}")
    
    return result

def show_dashboard(user_email, auth_service, data_service, model_service, prediction_service):
    # Sidebar for User Info & Logout
    st.sidebar.title("🩺 Control Panel")
    st.sidebar.info(f"Radiologist: **{user_email}**")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["📋 Diagnosis", "📊 Analytics"],
        key="nav_radio"
    )
    
    if st.sidebar.button("Logout"):
        auth_service.logout()
        st.rerun()
    
    # Import analytics here to avoid circular imports
    from ui.analytics import show_analytics, show_prediction_analytics
    
    # Route to selected page
    if page == "📋 Diagnosis":
        show_diagnosis_page(user_email, auth_service, data_service, model_service, prediction_service)
    else:
        show_analytics(data_service, user_email)
        show_prediction_analytics(prediction_service)

def show_diagnosis_page(user_email, auth_service, data_service, model_service, prediction_service):
    st.title("Clinical Diagnostic Dashboard")

    if not model_service.calibrated:
        st.warning("Calibration file not found (outputs/calibration.json). Probabilities are uncalibrated "
                   "and every prediction will be marked for clinician review.")

    # SECTION: New Analysis
    st.header("New Analysis")
    
    # Tabs for single and batch upload
    tab1, tab2 = st.tabs(["📄 Single Diagnosis", "📦 Batch Upload"])
    
    # TAB 1: Single Diagnosis
    with tab1:
        with st.expander("Process New X-ray", expanded=True):
            patient_name = st.text_input("Patient Name", placeholder="e.g., P-1024", key="single_patient")
            uploaded_file = st.file_uploader("Upload Chest X-ray", type=['jpg', 'jpeg', 'png'], key="single_upload")

            if uploaded_file and patient_name:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Scan", use_container_width=True)
                
                # Results live in session_state so later buttons (Generate report, PDF) don't erase them
                upload_key = (getattr(uploaded_file, 'file_id', uploaded_file.name), patient_name)
                
                if st.button("Run AI Diagnosis"):
                    with st.spinner("Analyzing anatomical features..."):
                        # Call the Model Service
                        result = run_prediction(model_service, prediction_service, image)
                        st.session_state['last_result'] = result
                        st.session_state['last_image'] = image
                        st.session_state['last_patient_name'] = patient_name
                        st.session_state['last_upload_key'] = upload_key
                        st.session_state['last_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state['last_heatmap'] = None
                        st.session_state['last_report'] = None
                        
                        # Save the result to Supabase
                        try:
                            data_service.save_diagnosis(user_email, result, patient_name)
                        except Exception as e:
                            st.error(f"Could not save to diagnostic history: {str(e)}")
                    
                    with st.spinner("Generating interpretation map (this may take 5-10 seconds)..."):
                        try:
                            # Pass the result to avoid redundant prediction
                            st.session_state['last_heatmap'] = model_service.generate_heatmap(image, result)
                        except Exception as e:
                            st.warning(f"Could not generate heatmap: {str(e)}")
                    
                    st.success("Analysis Complete!")
                
                if st.session_state.get('last_upload_key') == upload_key:
                    show_single_result(user_email, model_service, image)
    
    # TAB 2: Batch Upload
    with tab2:
        st.subheader("Process Multiple X-rays")
        st.info("Upload multiple X-ray images with their patient IDs to process them in batch.")
        
        # Batch upload section
        with st.form("batch_form"):
            # CSV input for patient IDs
            st.write("**Step 1:** Enter Patient IDs (one per line)")
            batch_patient_ids = st.text_area(
                "Patient IDs",
                placeholder="P-1001\nP-1002\nP-1003",
                height=100,
                key="batch_ids"
            )
            
            # Multiple file upload
            st.write("**Step 2:** Upload corresponding X-ray images")
            uploaded_files = st.file_uploader(
                "Upload multiple X-rays",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key="batch_upload"
            )
            
            submit_batch = st.form_submit_button("🚀 Process Batch", use_container_width=True)
        
        if submit_batch:
            patient_ids = [id.strip() for id in batch_patient_ids.strip().split('\n') if id.strip()]
            
            if not patient_ids or not uploaded_files:
                st.error("Please provide both patient IDs and X-ray images")
            elif len(patient_ids) != len(uploaded_files):
                st.error(f"Mismatch: {len(patient_ids)} patient IDs provided but {len(uploaded_files)} files uploaded")
            else:
                st.subheader("Processing Results")
                results = []
                
                # Process each file
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                results_container = st.container()
                
                for idx, (patient_id, uploaded_file) in enumerate(zip(patient_ids, uploaded_files)):
                    status_placeholder.write(f"Processing {idx + 1}/{len(uploaded_files)}: {patient_id}")
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    try:
                        # Open and predict
                        image = Image.open(uploaded_file)
                        result = run_prediction(model_service, prediction_service, image)
                        
                        # Save to database
                        data_service.save_diagnosis(user_email, result, patient_id)
                        
                        results.append({
                            'Patient ID': patient_id,
                            'Diagnosis': result['diagnosis'],
                            'Confidence': f"{result['confidence']*100:.1f}%",
                            'Triage': get_status_label(result['status']),
                            'Status': '✅ Success'
                        })
                    except Exception as e:
                        results.append({
                            'Patient ID': patient_id,
                            'Diagnosis': 'Error',
                            'Confidence': 'N/A',
                            'Triage': 'N/A',
                            'Status': f'❌ {str(e)[:30]}'
                        })
                
                status_placeholder.empty()
                progress_bar.empty()
                
                # Display results table
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Summary statistics
                successful = len([r for r in results if r['Status'] == '✅ Success'])
                referred = len([r for r in results if r['Triage'] == get_status_label(STATUS_REFER)])
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Processed", len(results))
                with col2:
                    st.metric("Successful", successful)
                with col3:
                    st.metric("Failed", len(results) - successful)
                with col4:
                    st.metric("Referred for Review", referred)

                st.success(f"Batch processing complete! {successful}/{len(results)} files processed successfully.")
                # No st.rerun() here: it wiped this table, and the history below is fetched after the batch anyway

    # SECTION: History Table
    st.divider()
    st.header("Past Diagnostic History")
    history = data_service.get_history(user_email)
    
    if history.data:
        # Add severity badge to history data
        df = pd.DataFrame(history.data)
        df['Severity'] = df['prediction'].apply(lambda x: get_severity_badge(x)['level'])
        
        # Format timestamps to readable format
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Search and Filter Section
        st.subheader("Search & Filter")
        search_col, diagnosis_col = st.columns(2)
        
        with search_col:
            search_patient = st.text_input("🔍 Search Patient ID", placeholder="e.g., P-1025")
        
        with diagnosis_col:
            diagnoses = ['All'] + df['prediction'].unique().tolist()
            selected_diagnosis = st.selectbox("Filter by Diagnosis", diagnoses)
        
        # Apply filters
        filtered_df = df.copy()
        
        if search_patient:
            filtered_df = filtered_df[filtered_df['patient_name'].str.contains(search_patient, case=False, na=False)]
        
        if selected_diagnosis != 'All':
            filtered_df = filtered_df[filtered_df['prediction'] == selected_diagnosis]
        
        # Display filtered history with download buttons
        st.subheader(f"Results: {len(filtered_df)} record(s)")
        
        # Create table header
        col1, col2, col3, col4, col5, col6, col7 = st.columns([1.5, 1.5, 2, 1.2, 1.5, 1.3, 1.5])
        with col1:
            st.write("**Patient ID**")
        with col2:
            st.write("**Diagnosis**")
        with col3:
            st.write("**Date & Time**")
        with col4:
            st.write("**Confidence**")
        with col5:
            st.write("**Severity**")
        with col6:
            st.write("**Triage**")
        with col7:
            st.write("**Action**")
        
        st.divider()
        
        # Display each record with download button in same row
        for idx, (_, record) in enumerate(filtered_df.iterrows()):
            col1, col2, col3, col4, col5, col6, col7 = st.columns([1.5, 1.5, 2, 1.2, 1.5, 1.3, 1.5])
            # Records saved before triage existed have no status (NaN in the DataFrame)
            status = record.get('status') if isinstance(record.get('status'), str) else None
            
            with col1:
                st.write(record['patient_name'])
            with col2:
                st.write(record['prediction'])
            with col3:
                st.write(record['created_at'])
            with col4:
                st.write(f"{record.get('confidence_score', 'N/A')}%")
            with col5:
                st.write(record['Severity'])
            with col6:
                st.write(get_status_label(status) if status else "—")
            with col7:
                if st.button("📥 Download", key=f"download_{idx}", use_container_width=True):
                    pdf_generator = PDFReportGenerator()
                    probabilities = {
                        'Normal': 0.33,
                        'Pneumonia': 0.33,
                        'Tuberculosis': 0.34
                    }
                    pdf_data = pdf_generator.generate_diagnosis_report(
                        patient_name=record.get('patient_name', 'Unknown'),
                        diagnosis=record.get('prediction', 'Unknown'),
                        confidence=record.get('confidence_score', 0),
                        probabilities=probabilities,
                        radiologist_email=record.get('radiologist_email', user_email),
                        timestamp=record.get('created_at', 'Unknown'),
                        status=status
                    )
                    st.download_button(
                        label="Click to Download PDF",
                        data=pdf_data,
                        file_name=f"Report_{record.get('patient_name')}_{record.get('created_at', '').replace(' ', '_').replace(':', '')}.pdf",
                        mime="application/pdf",
                        key=f"btn_{idx}",
                        use_container_width=True
                    )
    else:
        st.info("No records found in your portal.")