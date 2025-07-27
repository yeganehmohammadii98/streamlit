import streamlit as st
from PIL import Image


def main():
    st.title("🚀 Basic Streamlit Test")
    st.write("Testing basic deployment without Tesseract")

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
            st.success("✅ Image uploaded successfully!")

            # Show image info
            st.info(f"📏 Image size: {image.size}")
            st.info(f"🎨 Image mode: {image.mode}")

        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")

    st.write("If you can see this, basic Streamlit deployment is working! 🎉")


if __name__ == "__main__":
    main()