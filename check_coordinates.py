import inspect
import streamlit_image_coordinates

try:
    source_code = inspect.getsource(streamlit_image_coordinates.streamlit_image_coordinates)
    print("=== streamlit_image_coordinates.streamlit_image_coordinates SOURCE ===")
    print(source_code)
except Exception as e:
    import traceback
    traceback.print_exc()
