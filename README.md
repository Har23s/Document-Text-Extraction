This script uses the DeepSeek-OCR model from Hugging Face to convert document images into readable text. It runs entirely on CPU by disabling CUDA and mixed-precision operations for compatibility.

The script first scans the images/ folder for .jpg, .jpeg, or .png files and prompts the user to select one. It then loads the DeepSeek-OCR model and tokenizer, prepares a text-based prompt, and performs inference to extract text from the selected image.

The recognized text and visualization results are saved in the output/ directory. The code ensures stable performance on any system by forcing all computations to use float32.

In short, this script provides a step-by-step, CPU-compatible OCR pipeline that converts document images into readable text using the DeepSeek-AI model.
