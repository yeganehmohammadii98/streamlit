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


def main():
    st.title("🔍 Tesseract OCR Test App")
    st.write("Upload an image to extract text using Tesseract OCR")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp']
    )

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Show basic image info first
            st.info(f"📏 Image size: {image.size} | 🎨 Mode: {image.mode}")

            # Only show OCR option if Tesseract is available
            if TESSERACT_AVAILABLE:
                # Add a button to trigger OCR
                if st.button("Extract Text with Tesseract"):
                    with st.spinner("Processing image..."):
                        try:
                            # Test Tesseract availability first
                            try:
                                tesseract_version = pytesseract.get_tesseract_version()
                                st.info(f"🔧 Using Tesseract version: {tesseract_version}")
                            except Exception as version_error:
                                st.warning(f"⚠️ Could not get Tesseract version: {version_error}")

                            # Extract text using Tesseract
                            extracted_text = pytesseract.image_to_string(image)

                            # Display results
                            st.success("✅ Text extraction completed!")
                            st.subheader("Extracted Text:")

                            if extracted_text.strip():
                                st.text_area("Result", extracted_text, height=200)

                                # Show character and word count
                                char_count = len(extracted_text)
                                word_count = len(extracted_text.split())
                                st.info(f"📊 Characters: {char_count} | Words: {word_count}")
                            else:
                                st.warning("⚠️ No text found in the image")

                        except Exception as e:
                            st.error(f"❌ OCR Error: {str(e)}")
                            st.info("💡 This might indicate that Tesseract is not properly installed on the server")
            else:
                st.warning("🚫 Tesseract OCR is not available. Only image display is possible.")
                st.info("💡 To enable OCR, ensure pytesseract is installed and Tesseract system package is available.")

        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")

    # Add some info about the app
    with st.expander("ℹ️ About this app"):
        st.write("""
        This app tests Tesseract OCR functionality in Streamlit Cloud deployment.

        **How it works:**
        1. Upload an image containing text
        2. Click 'Extract Text with Tesseract'
        3. View the extracted text results

        **Supported formats:** PNG, JPG, JPEG, GIF, BMP

        **Note:** If you see errors, it might mean Tesseract isn't installed on the deployment server.
        """)

    # System info (for debugging)
    if st.checkbox("🔧 Show System Info (Debug)"):
        if TESSERACT_AVAILABLE:
            try:
                tesseract_version = pytesseract.get_tesseract_version()
                st.success(f"✅ Tesseract version: {tesseract_version}")
            except Exception as e:
                st.error(f"❌ Tesseract not accessible: {e}")
        else:
            st.error("❌ pytesseract not imported")

        try:
            import platform
            st.info(f"🖥️ Platform: {platform.system()} {platform.release()}")
        except:
            pass


if __name__ == "__main__":
    main()