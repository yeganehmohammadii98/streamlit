import streamlit as st
import pytesseract
from PIL import Image
import io


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
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Add a button to trigger OCR
            if st.button("Extract Text with Tesseract"):
                with st.spinner("Processing image..."):
                    try:
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
        try:
            tesseract_version = pytesseract.get_tesseract_version()
            st.success(f"✅ Tesseract version: {tesseract_version}")
        except:
            st.error("❌ Tesseract not found or not accessible")

        try:
            import platform
            st.info(f"🖥️ Platform: {platform.system()} {platform.release()}")
        except:
            pass


if __name__ == "__main__":
    main()