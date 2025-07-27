import streamlit as st
from PIL import Image
import io

# Try to import pytesseract with error handling
try:
    import pytesseract

    TESSERACT_AVAILABLE = True
    st.success("✅ Tesseract imported successfully!")
except ImportError as e:
    TESSERACT_AVAILABLE = False
    st.error(f"❌ Failed to import pytesseract: {e}")
except Exception as e:
    TESSERACT_AVAILABLE = False
    st.error(f"❌ Unexpected error importing pytesseract: {e}")


# Add this function to your existing app.py file, after the other page functions

def tesseract_test_page():
    """Simple Tesseract test page to verify OCR functionality"""
    st.header("🔍 Tesseract OCR Test")
    st.write("Test Tesseract installation and basic OCR functionality")

    # Try to import pytesseract with error handling
    try:
        import pytesseract
        TESSERACT_AVAILABLE = True
        st.success("✅ pytesseract imported successfully!")
    except ImportError as e:
        TESSERACT_AVAILABLE = False
        st.error(f"❌ Failed to import pytesseract: {e}")
        return
    except Exception as e:
        TESSERACT_AVAILABLE = False
        st.error(f"❌ Unexpected error importing pytesseract: {e}")
        return

    # Test Tesseract binary availability
    try:
        version = pytesseract.get_tesseract_version()
        st.success(f"✅ Tesseract binary found! Version: {version}")
    except Exception as e:
        st.error(f"❌ Tesseract binary not accessible: {e}")
        st.info("💡 Make sure packages.txt contains tesseract-ocr and tesseract-ocr-eng")
        return

    # File uploader for testing
    st.subheader("📤 Test Image Upload")
    uploaded_file = st.file_uploader(
        "Choose an image file for OCR testing",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp']
    )

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Show basic image info
            st.info(f"📏 Image size: {image.size} | 🎨 Mode: {image.mode}")

            # Add a button to trigger OCR
            if st.button("🔍 Extract Text with Tesseract"):
                with st.spinner("Processing image with OCR..."):
                    try:
                        # Extract text using Tesseract
                        extracted_text = pytesseract.image_to_string(image)

                        # Display results
                        st.success("✅ Text extraction completed!")
                        st.subheader("Extracted Text:")

                        if extracted_text.strip():
                            st.text_area("OCR Result:", extracted_text, height=200)

                            # Show statistics
                            char_count = len(extracted_text)
                            word_count = len(extracted_text.split())
                            st.info(f"📊 Characters: {char_count} | Words: {word_count}")
                        else:
                            st.warning("⚠️ No text found in the image")

                    except Exception as e:
                        st.error(f"❌ OCR Error: {str(e)}")
                        st.info("💡 This indicates Tesseract system package is not properly installed")

        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")

    # System debug info
    if st.checkbox("🔧 Show Debug Info"):
        try:
            import platform
            import os
            import shutil

            st.write("**System Information:**")
            st.write(f"Platform: {platform.system()} {platform.release()}")
            st.write(f"Python: {platform.python_version()}")

            # Check tesseract paths
            st.write("**Tesseract Search:**")
            tesseract_which = shutil.which('tesseract')
            st.write(f"which tesseract: {tesseract_which}")

            common_paths = [
                '/usr/bin/tesseract',
                '/usr/local/bin/tesseract',
                '/app/.apt/usr/bin/tesseract'
            ]

            st.write("**Common paths check:**")
            for path in common_paths:
                exists = os.path.exists(path)
                st.write(f"{path}: {'✅' if exists else '❌'}")

        except Exception as e:
            st.write(f"Debug info error: {e}")


# Also modify your main() function to include the new test page
# Update the sidebar navigation in your main() function:

def main():
    # Initialize database
    engine, SessionLocal = initialize_database()

    # App header
    st.title("🤖 Smart Invoice AI System")
    st.markdown("### Upload invoices and extract key information automatically")

    # Sidebar for navigation
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["Upload & Process", "View History", "Feedback & Corrections",
             "AI Learning Dashboard", "Model Statistics", "Settings",
             "🔍 Tesseract Test"]  # Add this line
        )

    # Main content based on selected page
    if page == "Upload & Process":
        upload_page()
    elif page == "View History":
        history_page()
    elif page == "Feedback & Corrections":
        show_feedback_history()
    elif page == "AI Learning Dashboard":
        show_learning_dashboard()
    elif page == "Model Statistics":
        stats_page()
    elif page == "Settings":
        settings_page()
    elif page == "🔍 Tesseract Test":  # Add this condition
        tesseract_test_page()
if __name__ == "__main__":
    main()