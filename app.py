import streamlit as st
import numpy as np
import cv2
from bus_people_counter import BusPeopleCounter

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Bus People Counter",
        page_icon=":bus:",
        layout="wide"
    )
    
    # Title and description
    st.title("🚌 Bus People Counter")
    st.write("Upload an image to count the number of people in a bus")
    
    counter = BusPeopleCounter()
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image of a bus to count people"
    )
    
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, channels="BGR")
        
        # Count people when button is clicked
        if st.button("Count People", type="primary"):
            # Process the image
            results = counter.count_people_in_bus(image)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("People Count")
                st.metric(
                    label="Total People Detected",
                    value=results['total_people']
                )
            
            # Display annotated image
            st.subheader("Annotated Image")
            st.image(
                results['annotated_image'], 
                channels="BGR"
            )


if __name__ == "__main__":
    main()