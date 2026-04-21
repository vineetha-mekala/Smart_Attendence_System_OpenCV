# # app.py
# import streamlit as st
# import cv2
# import numpy as np
# import os
# from system_process import recognize_faces_in_image
# from attendance_manager import mark_attendance

# st.set_page_config(page_title="Smart Attendance (MTCNN+FaceNet)", layout="wide")
# st.title("Smart Attendance — MTCNN + FaceNet Demo")

# st.write("Upload 1–3 classroom images (left to right, or columns). Process will detect faces, match with known students and save annotated images to `processed_images/`.")

# uploaded = st.file_uploader("Upload 1–3 photos", type=["jpg","jpeg","png"], accept_multiple_files=True)

# threshold = st.slider("Recognition similarity threshold (cosine)", 0.5, 0.95, 0.80, 0.01)

# if st.button("Process Attendance"):
#     if not uploaded:
#         st.warning("Please upload at least one image.")
#     else:
#         all_recognized = []
#         col1, col2 = st.columns(2)
#         for i, file in enumerate(uploaded):
#             bytes_data = file.read()
#             np_img = np.frombuffer(bytes_data, np.uint8)
#             img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#             # save temp
#             temp_name = f"temp_{i}.jpg"
#             cv2.imwrite(temp_name, img)

#             recognized, unknowns, annotated, save_path = recognize_faces_in_image(temp_name, save_name=file.name, threshold=threshold)

#             all_recognized.extend(recognized)

#             # Display
#             with col1:
#                 st.subheader(f"Processed: {file.name}")
#                 st.image(annotated[:,:,::-1], use_column_width=True)
#             with col2:
#                 st.write("Recognized:")
#                 st.write(list(set(recognized)))
#                 st.write(f"Unknown faces: {len(unknowns)}")
#                 st.write(f"Saved to: `{save_path}`")

#             # cleanup temp
#             try:
#                 os.remove(temp_name)
#             except:
#                 pass

#         unique_names = list(set(all_recognized))
#         st.success(f"Attendance marked for {len(unique_names)} students.")
#         df = mark_attendance(unique_names)
#         st.dataframe(df)
#         st.markdown(f"Download attendance CSV: `attendance.csv` (saved in project folder)")

# app.py
import streamlit as st
import cv2
import numpy as np
import os
from system_process import recognize_faces_in_image
from attendance_manager import mark_attendance

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title="Smart Attendance (MTCNN+FaceNet)", layout="wide")
st.title("Smart Attendance — MTCNN + FaceNet Demo")

# -------------------------
# Initialize session state
# -------------------------
if "webcam_photo" not in st.session_state:
    st.session_state["webcam_photo"] = None

if "show_webcam" not in st.session_state:
    st.session_state["show_webcam"] = False

st.write("""
Upload 1–3 classroom images OR capture an image using the webcam.
System detects all faces, recognizes students, and marks attendance.
""")

# -------------------------
# File Upload Section
# -------------------------
uploaded = st.file_uploader(
    "Upload 1–3 photos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# -------------------------
# Webcam Capture Section
# -------------------------
st.subheader("Or capture a classroom photo using your webcam")

# Button to open webcam
if st.button("Use Webcam to Capture Photo"):
    st.session_state["show_webcam"] = True

# Show webcam only when triggered
if st.session_state["show_webcam"] and st.session_state["webcam_photo"] is None:
    webcam_img = st.camera_input("Take Photo")

    if webcam_img is not None:
        st.session_state["webcam_photo"] = webcam_img
        st.session_state["show_webcam"] = False
        #st.experimental_rerun()
        st.rerun()

# Show captured photo preview
elif st.session_state["webcam_photo"] is not None:
    st.image(st.session_state["webcam_photo"], caption="Captured Photo", width=300)

    colA, colB = st.columns(2)

    # Retake photo
    if colA.button("Retake Photo"):
        st.session_state["webcam_photo"] = None
        st.session_state["show_webcam"] = False
        #st.experimental_rerun()
        st.rerun()

    colB.success("Photo Ready ✔")

# -------------------------
# Recognition Threshold Slider
# -------------------------
threshold = st.slider(
    "Recognition similarity threshold (cosine)",
    0.5, 0.95, 0.80, 0.01
)

# -------------------------
# Process Attendance Button
# -------------------------
if st.button("Process Attendance"):

    # Combine all input images (uploaded + webcam)
    input_images = []

    if uploaded:
        input_images.extend(uploaded)

    if st.session_state["webcam_photo"] is not None:
        input_images.append(st.session_state["webcam_photo"])

    if not input_images:
        st.warning("Please upload or capture at least one image.")
        st.stop()

    all_recognized = []
    col1, col2 = st.columns(2)

    # Process each image
    for i, file in enumerate(input_images):

        # Convert Bytes → OpenCV Image
        bytes_data = file.getvalue()
        np_img = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Save temporary image
        temp_name = f"temp_{i}.jpg"
        cv2.imwrite(temp_name, img)

        # Run recognition
        recognized, unknowns, annotated, save_path = recognize_faces_in_image(
            temp_name,
            save_name=f"processed_{i}.jpg",
            threshold=threshold
        )

        all_recognized.extend(recognized)

        # Show processed results
        with col1:
            st.subheader(f"Processed Image {i+1}")
            st.image(annotated[:, :, ::-1], use_column_width=True)

        with col2:
            st.write("Recognized Students:")
            st.write(list(set(recognized)))
            st.write(f"Unknown Faces: {len(unknowns)}")
            st.write(f"Saved at: `{save_path}`")

        # Remove temp
        try:
            os.remove(temp_name)
        except:
            pass

    # Final attendance list
    unique_names = list(set(all_recognized))

    st.success(f"Attendance marked for {len(unique_names)} students.")

    # Update attendance CSV
    df = mark_attendance(unique_names)
    st.dataframe(df)

    st.markdown("Download attendance CSV: `attendance.csv` (saved in project folder)")

