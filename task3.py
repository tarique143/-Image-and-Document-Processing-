import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import easyocr
import torch  # For GPU detection

# --- Configuration ---
st.set_page_config(
    page_title="🍏 Food & Ingredients Bill Processor",
    page_icon="🍏",
    layout="wide"
)

# Initialize OCR reader (only once using st.cache_resource)
@st.cache_resource
def load_ocr_reader():
    try:
        # Check if GPU is available
        gpu_available = torch.cuda.is_available()
        return easyocr.Reader(['en'], gpu=gpu_available)
    except Exception as e:
        st.error(f"OCR initialization failed: {str(e)}")
        return None

# --- Core Functions ---
def generate_ai_description(image_array):
    """Generate dynamic AI description based on image content"""
    try:
        # Get detected objects (using your existing function)
        detections = detect_objects(image_array, detection_mode)
        
        # Get OCR text (using your existing function)
        ocr_text = perform_ocr(image_array)
        
        # Analyze the content
        food_items = [obj['label'] for obj in detections if obj['label'] in ['Apple', 'Flour']]
        has_receipt = any(obj['label'] == 'Total' for obj in detections)
        
        # Extract key info from OCR
        has_total = "total" in ocr_text.lower()
        has_dates = any(word in ocr_text.lower() for word in ["expiry", "date"])
        
        # Build description dynamically
        description_parts = []
        
        if food_items:
            description_parts.append(f"Found food items: {', '.join(food_items)}")
        
        if has_receipt or has_total:
            description_parts.append("Receipt detected with purchase details")
            if has_total:
                # Extract the total line if available
                total_line = next((line for line in ocr_text.split('\n') 
                                if 'total' in line.lower()), None)
                if total_line:
                    description_parts.append(f"Purchase total: {total_line.split()[-1]}")
        
        if has_dates:
            description_parts.append("Expiry/date information found")
        
        if not description_parts:
            # Fallback description if nothing specific detected
            if len(ocr_text) > 20:
                return "Document with text content detected"
            return "Image contains food-related items"
        
        # Add context about image quality
        if len(ocr_text) < 10 and not detections:
            description_parts.append("(Low detail or blurry image detected)")
        
        return ". ".join(description_parts) + "."
    
    except Exception as e:
        st.error(f"Description generation error: {str(e)}")
        return "AI description unavailable"
    
def detect_objects(image, mode):
    """Mock object detection for food/ingredients/bills"""
    height, width = image.shape[:2]
    if mode == "Food Items":
        return [{"label": "Apple", "confidence": 0.92, "bbox": [width//4, height//4, 100, 100]}]
    elif mode == "Ingredients":
        return [{"label": "Flour", "confidence": 0.85, "bbox": [width//2, height//2, 90, 90]}]
    else:  # Receipts
        return [{"label": "Total", "confidence": 0.95, "bbox": [width//6, height//6, 200, 50]}]

def perform_ocr(image_array):
    """Real OCR implementation using EasyOCR"""
    try:
        # Convert to BGR format
        img_bgr = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)
        
        # Initialize reader if not already done
        if 'reader' not in st.session_state:
            st.session_state.reader = load_ocr_reader()
        
        if st.session_state.reader is None:
            return "OCR engine failed to initialize"
        
        # Perform OCR
        results = st.session_state.reader.readtext(img_bgr)
        
        # Format results
        return "\n".join([f"{text} ({confidence:.0%})" for (_, text, confidence) in results])
    
    except Exception as e:
        st.error(f"OCR processing error: {str(e)}")
        return ""

def extract_insights(text):
    """Extract key info from OCR text"""
    insights = []
    text_lower = text.lower()
    
    if "total" in text_lower:
        insights.append("💰 Total amount detected")
    if any(word in text_lower for word in ["expiry", "date"]):
        insights.append("📅 Expiry date mentioned")
    return insights or ["🔍 No key insights automatically detected"]

# --- UI Layout ---
st.title("🍏 Food & Ingredients Bill Processor")
st.markdown("Upload food/ingredient images or receipts for AI analysis and OCR extraction")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    detection_mode = st.selectbox(
        "Detection Focus",
        ["Food Items", "Ingredients", "Receipts"]
    )

# Main Content
uploaded_file = st.file_uploader(
    "📤 Upload Food/Receipt Image (JPG/PNG)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True)
        
        st.subheader("🧠 AI Description")
        st.info(generate_ai_description(img_array))
    
    with col2:
        st.subheader("🔍 Detected Items")
        detections = detect_objects(img_array, detection_mode)
        
        if detections:
            fig, ax = plt.subplots()
            ax.imshow(image)
            for obj in detections:
                x,y,w,h = obj["bbox"]
                rect = patches.Rectangle((x,y),w,h,linewidth=2,edgecolor='red',facecolor='none')
                ax.add_patch(rect)
                plt.text(x,y,f"{obj['label']} ({obj['confidence']:.0%})",
                        color='red', bbox=dict(facecolor='white', alpha=0.7))
            plt.axis('off')
            st.pyplot(fig)
        else:
            st.warning("No items detected")
        
        st.subheader("💡 Key Insights")
        text = perform_ocr(img_array)
        for insight in extract_insights(text):
            st.success(insight)
    
    with st.expander("📝 View Full OCR Text"):
        st.text(text)
else:
    st.info("Please upload an image to begin analysis")

# Footer
st.markdown("---")
st.caption("AI Food Processor v1.0 | Detects items, extracts text, and provides insights")