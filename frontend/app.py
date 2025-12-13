import streamlit as st
import requests
from PIL import Image
import io
import base64
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title="ChipSight - PCB Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme matching the mockup
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stApp {
        background-color: #0E1117;
    }
    h1 {
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .uploadedFile {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E1E1E;
        color: #FFFFFF;
        border-radius: 5px 5px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF0000;
        color: #FFFFFF;
    }
    .defect-info {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    .critical {
        border-left-color: #FF0000;
    }
    .moderate {
        border-left-color: #FFA500;
    }
    .minor {
        border-left-color: #00FF00;
    }
    </style>
    """, unsafe_allow_html=True)

# Backend API URL
BACKEND_URL = st.sidebar.text_input(
    "Backend API URL",
    value="https://chipsight-backend-837072298188.us-east1.run.app",
    help="URL of the FastAPI backend service"
)

# Title
st.title("🔍 ChipSight")

# Initialize session state
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "detections" not in st.session_state:
    st.session_state.detections = []
if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None
if "original_image" not in st.session_state:
    st.session_state.original_image = None

def get_severity_color(severity_level: str) -> str:
    """Get color for severity level"""
    color_map = {
        "Critical": "#FF0000",
        "Moderate": "#FFA500",
        "Minor": "#00FF00",
    }
    return color_map.get(severity_level, "#FFFFFF")

def format_class_name(class_name: str) -> str:
    """Format class name for display"""
    return class_name.replace("_", " ").title()

# File upload section
st.markdown("### Upload an image")
uploaded_file = st.file_uploader(
    "Drag and drop file here",
    type=["jpg", "jpeg", "png"],
    help="Upload a PCB image for defect detection. Max file size: 200MB",
    label_visibility="collapsed"
)

# Handle file upload
if uploaded_file is not None:
    # Store uploaded file
    st.session_state.uploaded_file = uploaded_file
    
    # Read and store original image
    image_bytes = uploaded_file.read()
    st.session_state.original_image = Image.open(io.BytesIO(image_bytes))
    
    # Display file info
    file_size_kb = len(image_bytes) / 1024
    col1, col2 = st.columns([10, 1])
    with col1:
        st.markdown(f"""
        <div class="uploadedFile">
            <strong>{uploaded_file.name}</strong> ({file_size_kb:.1f}KB)
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("✕", key="remove_file"):
            st.session_state.uploaded_file = None
            st.session_state.detections = []
            st.session_state.annotated_image = None
            st.session_state.original_image = None
            st.rerun()
    
    # Process image if not already processed
    if st.session_state.annotated_image is None or st.button("🔍 Analyze Image", type="primary"):
        with st.spinner("Analyzing image for defects..."):
            try:
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Send to backend
                files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
                response = requests.post(
                    f"{BACKEND_URL}/predict?return_image=true",
                    files=files,
                    timeout=(10, 300)
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.detections = data.get("detections", [])
                    
                    # Decode annotated image
                    if data.get("annotated_image_base64"):
                        img_data = base64.b64decode(data["annotated_image_base64"])
                        st.session_state.annotated_image = Image.open(io.BytesIO(img_data))
                    else:
                        st.session_state.annotated_image = st.session_state.original_image.copy()
                    
                    st.success(f"Found {len(st.session_state.detections)} defect(s)")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to backend at {BACKEND_URL}. Make sure the FastAPI server is running.")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    # Display results in tabs
    if st.session_state.original_image and st.session_state.annotated_image:
        tab1, tab2 = st.tabs(["Original", "Detected Defects"])
        
        with tab1:
            st.image(st.session_state.original_image, use_container_width=True)
        
        with tab2:
            # Create two columns for side-by-side display
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(st.session_state.annotated_image, use_container_width=True)
            
            with col2:
                st.markdown("### Defect Details")
                
                if st.session_state.detections:
                    # Sort by severity score (highest first)
                    sorted_detections = sorted(
                        st.session_state.detections,
                        key=lambda x: x["severity_score"],
                        reverse=True
                    )
                    
                    for i, det in enumerate(sorted_detections):
                        severity = det["severity_level"]
                        color = get_severity_color(severity)
                        class_name = format_class_name(det["class_name"])
                        confidence = int(det["confidence"] * 100)
                        
                        # Create defect info box
                        severity_class = severity.lower()
                        st.markdown(f"""
                        <div class="defect-info {severity_class}">
                            <h4 style="color: {color}; margin: 0;">{class_name}</h4>
                            <p style="margin: 5px 0;"><strong>Confidence:</strong> {confidence}%</p>
                            <p style="margin: 5px 0; color: {color};"><strong>Severity:</strong> {severity}</p>
                            <p style="margin: 5px 0; font-size: 0.9em; color: #888;">Score: {det['severity_score']:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No defects detected in this image.")
        
        # Summary statistics
        if st.session_state.detections:
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            critical_count = sum(1 for d in st.session_state.detections if d["severity_level"] == "Critical")
            moderate_count = sum(1 for d in st.session_state.detections if d["severity_level"] == "Moderate")
            minor_count = sum(1 for d in st.session_state.detections if d["severity_level"] == "Minor")
            
            with col1:
                st.metric("Total Defects", len(st.session_state.detections))
            with col2:
                st.metric("Critical", critical_count, delta=None, delta_color="inverse")
            with col3:
                st.metric("Moderate", moderate_count)
            with col4:
                st.metric("Minor", minor_count)

else:
    st.info("Please upload a PCB image to begin defect detection.")
    
    # Show example or instructions
    st.markdown("""
    ### Supported Formats
    - JPG, JPEG, PNG
    - Maximum file size: 200MB
    """)

