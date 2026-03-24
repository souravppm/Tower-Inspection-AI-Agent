import streamlit as st
import requests
import pandas as pd
from PIL import Image, ImageDraw
import plotly.graph_objects as go
from fpdf import FPDF
import io
import pickle
import os
import base64

CACHE_DIR = ".cache"
CACHE_PATH = os.path.join(CACHE_DIR, "analysis_state.pkl")

def save_state():
    os.makedirs(CACHE_DIR, exist_ok=True)
    state_to_save = {
        "uploaded_filenames": st.session_state.uploaded_filenames,
        "raw_images": st.session_state.raw_images,
        "final_results": st.session_state.final_results
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(state_to_save, f)

def load_state():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def clear_all_state():
    if os.path.exists(CACHE_PATH):
        try:
            os.remove(CACHE_PATH)
        except Exception:
            pass
    st.session_state.final_results = None
    st.session_state.uploaded_filenames = []
    st.session_state.raw_images = []

@st.cache_data
def generate_llm_report(detections):
    try:
        rep_res = requests.post("http://127.0.0.1:8000/report", json={"detections": detections})
        if rep_res.status_code == 200:
            return rep_res.json().get("report", "No report generated.")
        return "Failed to generate report from backend."
    except Exception as e:
        return f"API error generating report: {e}"

def create_pdf_report(report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Fleet-Wide Executive Summary", ln=True, align='C')
    pdf.ln(10)
    
    # Remove markdown asterisks completely
    cleaned_text = report_text.replace('**', '').replace('*', '')
    
    # Sanitize text for FPDF basic fonts
    safe_text = cleaned_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_text)
    return pdf.output()

def main():
    st.set_page_config(page_title="Tower AI Inspector", layout="wide", page_icon="🗼")
    
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)
    
    # Initialization Cache (Top of script):
    if 'initialized_cache' not in st.session_state:
        st.session_state.initialized_cache = True
        loaded = load_state()
        if loaded:
            st.session_state.final_results = loaded.get("final_results", None)
            st.session_state.uploaded_filenames = loaded.get("uploaded_filenames", [])
            st.session_state.raw_images = loaded.get("raw_images", [])
        else:
            st.session_state.final_results = None
            st.session_state.uploaded_filenames = []
            st.session_state.raw_images = []

    st.sidebar.title("⚙️ Model & Settings")
    st.sidebar.markdown("### Metadata")
    st.sidebar.metric("Model", "Custom YOLOv8")
    st.sidebar.metric("Training Accuracy (mAP)", "93%")
    st.sidebar.metric("Inference Time", "~3ms")
    st.sidebar.markdown("---")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    
    if st.sidebar.button("🗑️ Reset All Sessions"):
        clear_all_state()
        st.rerun()

    st.title("Automated Structural Integrity Analysis")
    st.write("This system detects tower **joints** and **sides** from drone images to analyze structural health fleet-wide.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Batch file uploader
        uploaded_files = st.file_uploader("Upload tower images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_files:
            current_filenames = sorted([f.name for f in uploaded_files])
            if current_filenames != st.session_state.uploaded_filenames:
                # Clear state only on new upload list
                st.session_state.uploaded_filenames = current_filenames
                st.session_state.final_results = None
                
                raws = []
                for f in uploaded_files:
                    f.seek(0)
                    raws.append({
                        "filename": f.name, 
                        "image_bytes": f.getvalue(),
                        "file_type": f.type
                    })
                st.session_state.raw_images = raws
                save_state()
                st.rerun()

        if not uploaded_files and st.session_state.raw_images:
            st.info(f"Loaded {len(st.session_state.raw_images)} images from previous session.")

        # Analyze Button handling
        analyze_button = False
        if st.session_state.raw_images and st.session_state.final_results is None:
            analyze_button = st.button("Analyze Tower Images", type="primary", use_container_width=True)

        if st.session_state.final_results is not None:
            if st.button("🔄 Clear Analysis", use_container_width=True):
                st.session_state.final_results = None
                save_state()
                st.rerun()
            
    with col2:
        # The Logic (Inside the Button):
        if analyze_button:
            with st.spinner("Analyzing with YOLOv8 Vision Model & Agentic AI..."):
                
                files_payload = [
                    ('files', (img["filename"], img["image_bytes"], img.get("file_type", "image/jpeg"))) 
                    for img in st.session_state.raw_images
                ]
                
                try:
                    response = requests.post("http://localhost:8000/detect/", files=files_payload)
                    if response.status_code != 200:
                        try:
                            err_data = response.json()
                            err_detail = err_data.get("detail", "")
                            if "OOD_ERROR" in err_detail:
                                st.error(err_detail)
                            else:
                                st.error(f"Backend API Error: {err_detail or response.text}")
                        except Exception:
                            st.error(f"Backend API Error: {response.text}")
                        st.stop()
                        
                    result = response.json()
                    annotated_images_b64 = result.get("annotated_images", {})
                    all_detections = result.get("detections", [])
                except Exception as e:
                    st.error(f"Failed to connect to backend: {str(e)}")
                    st.stop()
                
                # Consolidated LLM Report
                llm_summary = "No valid detections found across uploaded images."
                if all_detections:
                    llm_summary = generate_llm_report(all_detections)
                        
                # Create PDF report buffer during analysis
                pdf_bytes = create_pdf_report(llm_summary)
                
                # Decode base64 images
                annotated_images_decoded = {}
                for filename, b64_str in annotated_images_b64.items():
                    try:
                        annotated_images_decoded[filename] = base64.b64decode(b64_str)
                    except Exception as e:
                        pass
                
                # SAVE base data into a single dictionary
                st.session_state.final_results = {
                    "all_detections": all_detections,
                    "annotated_images": annotated_images_decoded,
                    "llm_report": llm_summary,
                    "pdf_bytes": pdf_bytes
                }
                save_state()
                st.rerun()

    # The Rendering (OUTSIDE the Button):
    if st.session_state.final_results:
        res = st.session_state.final_results
        
        # Calculate health score dynamically based on slider
        all_detections = res["all_detections"]
        joints = [d for d in all_detections if d.get("class_name", "").lower() == "joint" and d.get("confidence", 0) >= confidence_threshold]
        total_joints = len(joints)
        if joints:
            avg_conf = sum(d.get("confidence", 0) for d in joints) / total_joints
        else:
            avg_conf = 0.0
        health_score = int(avg_conf * 100)
        
        annotated_images_dict = res.get("annotated_images", {})
        
        st.success("Fleet Analysis Complete!")
        
        # Enterprise Dashboard Layout
        db_col1, db_col2 = st.columns([1.5, 1])
        
        with db_col1:
            # Images
            if annotated_images_dict:
                tabs = st.tabs(list(annotated_images_dict.keys()))
                for tab, filename in zip(tabs, annotated_images_dict.keys()):
                    with tab:
                        st.image(annotated_images_dict[filename], caption=f"Annotated: {filename}", use_container_width=True)
            else:
                st.warning("No annotated images were returned from the analysis.")
            
            # LLM Report & PDF
            with st.expander('📄 View Detailed Executive Report', expanded=True):
                st.markdown("### Fleet-Wide Executive Summary")
                st.write(res["llm_report"])
            
            st.download_button(
                label="📥 Download Official PDF Report",
                data=bytes(res["pdf_bytes"]),
                file_name="fleet_wide_report.pdf",
                mime="application/pdf"
            )

        with db_col2:
            # Metrics
            met1, met2 = st.columns(2)
            met1.metric(label="Detected Joints", value=total_joints)
            met2.metric(label="Average AI Confidence", value=f"{avg_conf:.1%}")

            # Gauge Chart
            st.markdown("### Overall Tower Health")
            if health_score < 50:
                color = "red"
            elif health_score <= 79:
                color = "gold"
            else:
                color = "green"
                
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = health_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fleet Health Score"},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(255, 0, 0, 0.1)'},
                        {'range': [50, 79], 'color': 'rgba(255, 215, 0, 0.1)'},
                        {'range': [80, 100], 'color': 'rgba(0, 128, 0, 0.1)'}
                    ],
                }
            ))
            fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=300)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Detection Details")
            if all_detections:
                display_data = []
                for d in all_detections:
                    display_data.append({
                        "Component": str(d.get('class_name', 'Unknown')).capitalize(),
                        "Confidence": d.get('confidence', 0.0),
                        "Damage (%)": d.get('damage_area_percentage', 0.0)
                    })
                df = pd.DataFrame(display_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No detections to display.")
    elif st.session_state.raw_images:
        tabs = st.tabs([img["filename"] for img in st.session_state.raw_images])
        for tab, img in zip(tabs, st.session_state.raw_images):
            with tab:
                base_image = Image.open(io.BytesIO(img["image_bytes"])).convert("RGB")
                st.image(base_image, caption=f"Uploaded: {img['filename']}", use_container_width=True)

if __name__ == "__main__":
    main()
