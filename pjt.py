'''from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import random
import base64
from io import BytesIO
output_path = './uploads'
import wave
from scipy.fftpack import fft, ifft

def encode_phase(audio_file, secret_message):
    message_bits = ''.join(format(ord(c), '08b') for c in secret_message) + '1111111111111110'  # End marker
    with wave.open(audio_file, "rb") as wav:
        params = wav.getparams()
        audio_frames = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)

    spectrum = fft(audio_frames)

    # Modify phase spectrum to embed data
    phase = np.angle(spectrum)  # Get phase
    for i, bit in enumerate(message_bits):
        phase[i] += np.pi if bit == '1' else -np.pi  # Modify phase slightly

    # Reconstruct audio using modified phase
    modified_spectrum = np.abs(spectrum) * np.exp(1j * phase)
    stego_audio = np.real(ifft(modified_spectrum)).astype(np.int16)

    buffer = BytesIO()
    with wave.open(buffer, "wb") as stego_wav:
        stego_wav.setparams(params)
        stego_wav.writeframes(stego_audio.tobytes())

    buffer.seek(0)
    encoded_audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return encoded_audio_base64


def decode_phase(stego_audio):
    with wave.open(stego_audio, "rb") as wav:
        audio_frames = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)

    # Apply FFT to extract phase
    spectrum = fft(audio_frames)
    phase = np.angle(spectrum)  # Extract phase

    # Convert phase shifts back to binary
    extracted_bits = ['1' if p > 0 else '0' for p in phase[:len(phase)//8]]

    # Convert binary to text
    extracted_message = ''.join(chr(int(''.join(extracted_bits[i:i+8]), 2)) for i in range(0, len(extracted_bits), 8))

    print("🔍 Extracted Message:", extracted_message.split('\x00')[0])  # Remove noise
    return extracted_message

def derive_seed(password):
    """Derives a seed from the password using SHA-256."""
    salt = hashlib.sha256(password.encode()).digest()
    seed = int.from_bytes(salt, 'big')
    return seed

def encrypt_message(message, key):
    """Encrypts the message using AES."""
    iv = get_random_bytes(AES.block_size)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(message.encode(), AES.block_size))
    return iv + ciphertext

def decrypt_message(ciphertext, key):
    """Decrypts the message using AES."""
    iv = ciphertext[:AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(ciphertext[AES.block_size:]), AES.block_size)
    return plaintext.decode()

def encode_message(image_path, message, password, output_path):
    print("hiiiiiiiiiiiiiiii")
    print(image_path)
    """Encodes a message into an image using LSB steganography with random RGB selection."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("The image could not be loaded.")

        key = hashlib.sha256(password.encode()).digest()  # AES key
        encrypted_message = encrypt_message(message, key)
        message_bits = ''.join([format(byte, '08b') for byte in encrypted_message])
        message_length = len(message_bits)
        length_bits = format(len(encrypted_message), '016b')  # 16 bits for encrypted message length

        max_capacity = image.size # only one bit per pixel.
        if (len(length_bits) + 16) > max_capacity: # +16 for length bits
            raise ValueError("Message is too long to be encoded in the given image.")

        combined_bits = length_bits + message_bits

        seed = derive_seed(password)
        random.seed(seed)

        flat_image = image.flatten().astype(np.int32) #convert to int32

        bit_index = 0
        for i in range(len(combined_bits)):
            channel = random.randint(0, 2)  # Randomly choose R, G, or B
            flat_image[bit_index + channel] = (flat_image[bit_index + channel] & ~1) | int(combined_bits[i])
            bit_index += 3

        encoded_image = flat_image.reshape(image.shape).astype(np.uint8) #convert back to uint8
        _, buffer = cv2.imencode('.png', encoded_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        print("Encoding completed and image prepared in base64.")
        return base64_image
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def decode_message(encoded_image_path, password):
    """Decodes a message from an encoded image with random RGB selection."""
    try:
        encoded_image = cv2.imread(encoded_image_path)
        if encoded_image is None:
            raise ValueError("The encoded image could not be loaded.")

        seed = derive_seed(password)
        random.seed(seed)

        flat_image = encoded_image.flatten()
        length_bits = ""
        bit_index = 0
        for _ in range(16):
            channel = random.randint(0, 2)
            length_bits += str(flat_image[bit_index + channel] & 1)
            bit_index += 3

        message_length = int(length_bits, 2)
        message_bits = ""

        for _ in range(message_length * 8):
            channel = random.randint(0, 2)
            message_bits += str(flat_image[bit_index + channel] & 1)
            bit_index += 3

        message_bytes = bytes([int(message_bits[i:i + 8], 2) for i in range(0, len(message_bits), 8)])
        key = hashlib.sha256(password.encode()).digest()
        decoded_message = decrypt_message(message_bytes, key)
        print(f"Decoded message: {decoded_message}")
        return decoded_message

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


app = Flask(__name__)
image_path = ""
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    global image_path
    if request.method == 'POST':
        mode = request.form.get('mode')
        operation = request.form.get('operation')
        password = request.form.get('password')
        secret_text = request.form.get('secret_text')
        file = request.files.get('file')
        print(mode)
        print(operation)
        print(password)
        print(secret_text)
        print(file)
        if mode == 'image': # Image mode
            if operation == 'encode':
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    base64_image = encode_message(filepath, secret_text, password, './uploads')

                    return jsonify({
                "success": True,
                "encoded_image_base64": base64_image
                })
                else:
                    return jsonify({'error': 'Invalid file or file type.'}), 400

            elif operation == 'decode':
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)

                    decoded_message = decode_message(filepath, password)
                    return jsonify({
                        "success": True,
                        "decoded_message": decoded_message,
                    })
                else:
                    return jsonify({'error': 'Invalid file or file type.'}), 400

        elif mode == 'audio': 
            if operation=='encode': # Audio mode
                if file:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    encoded_audio= encode_phase(filepath,secret_text)
                    return jsonify({
                'mode': mode,
                'message': 'Audio encoded successfully.',
                'filename': filename,
                'base64_audio': encoded_audio,
                'download_filename': 'encoded_audio.wav'
            })
            elif operation=='decode':
                if file:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    decoded_message = decode_phase(filepath)
                    return jsonify({
                        "success": True,
                        "decoded_message": decoded_message,
                    })
            else:
                return jsonify({'error': 'Audio file not uploaded.'}), 400

        return jsonify({'message': 'Request received, but no image or audio file provided.'}), 400

    return render_template('index.html')

@app.route('/encode', methods=['POST'])
def encode():
    return jsonify({"success": True, "encoded_image_url": image_path})


if __name__ == '__main__':
    app.run(debug=True)'''




from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import random
import base64
from io import BytesIO
import wave
from scipy.fftpack import fft, ifft

output_path = './uploads'

def encode_phase(audio_file, secret_message):
    """Encodes a message into an audio file using phase coding steganography."""
    message_bits = ''.join(format(ord(c), '08b') for c in secret_message) + '1111111111111110'  # End marker
    with wave.open(audio_file, "rb") as wav:
        params = wav.getparams()
        audio_frames = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)

    spectrum = fft(audio_frames)

    # Modify phase spectrum to embed data
    phase = np.angle(spectrum)  # Get phase
    for i, bit in enumerate(message_bits):
        if i >= len(phase):
            break  # Prevent index out of range
        phase[i] += np.pi if bit == '1' else -np.pi  # Modify phase slightly

    # Reconstruct audio using modified phase
    modified_spectrum = np.abs(spectrum) * np.exp(1j * phase)
    stego_audio = np.real(ifft(modified_spectrum)).astype(np.int16)

    buffer = BytesIO()
    with wave.open(buffer, "wb") as stego_wav:
        stego_wav.setparams(params)
        stego_wav.writeframes(stego_audio.tobytes())

    buffer.seek(0)
    encoded_audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return encoded_audio_base64


def decode_phase(stego_audio):
    """Decodes a message from an audio file using phase coding steganography."""
    with wave.open(stego_audio, "rb") as wav:
        audio_frames = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)

    # Apply FFT to extract phase
    spectrum = fft(audio_frames)
    phase = np.angle(spectrum)  # Extract phase

    # Convert phase shifts back to binary
    extracted_bits = ['1' if p > 0 else '0' for p in phase[:len(phase)//8]]

    # Convert binary to text
    try:
        extracted_message = ""
        for i in range(0, len(extracted_bits), 8):
            if i + 8 <= len(extracted_bits):
                byte = ''.join(extracted_bits[i:i+8])
                extracted_message += chr(int(byte, 2))
        
        # Look for terminator sequence
        if '1111111111111110' in ''.join(extracted_bits):
            end_index = extracted_message.find('\uffff')
            if end_index != -1:
                extracted_message = extracted_message[:end_index]
        
        return extracted_message.split('\x00')[0]  # Remove noise
    except Exception as e:
        print(f"Error decoding message: {e}")
        return "Error decoding message"


def derive_seed(password):
    """Derives a seed from the password using SHA-256."""
    salt = hashlib.sha256(password.encode()).digest()
    seed = int.from_bytes(salt, 'big')
    return seed


def encrypt_message(message, key):
    """Encrypts the message using AES."""
    iv = get_random_bytes(AES.block_size)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(message.encode(), AES.block_size))
    return iv + ciphertext


def decrypt_message(ciphertext, key):
    """Decrypts the message using AES."""
    try:
        iv = ciphertext[:AES.block_size]
        cipher = AES.new(key, AES.MODE_CBC, iv)
        plaintext = unpad(cipher.decrypt(ciphertext[AES.block_size:]), AES.block_size)
        return plaintext.decode()
    except Exception as e:
        print(f"Decryption error: {e}")
        return "Decryption failed. Incorrect password or corrupted data."


def encode_message(image_path, message, password, output_path):
    """Encodes a message into an image using LSB steganography with random RGB selection."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("The image could not be loaded.")

        key = hashlib.sha256(password.encode()).digest()  # AES key
        encrypted_message = encrypt_message(message, key)
        message_bits = ''.join([format(byte, '08b') for byte in encrypted_message])
        message_length = len(message_bits)
        length_bits = format(len(encrypted_message), '016b')  # 16 bits for encrypted message length

        max_capacity = image.size # only one bit per pixel.
        if (len(length_bits) + message_length) > max_capacity: # Check capacity
            raise ValueError("Message is too long to be encoded in the given image.")

        combined_bits = length_bits + message_bits

        seed = derive_seed(password)
        random.seed(seed)

        flat_image = image.flatten().astype(np.int32) #convert to int32

        bit_index = 0
        for i in range(len(combined_bits)):
            channel = random.randint(0, 2)  # Randomly choose R, G, or B
            flat_image[bit_index + channel] = (flat_image[bit_index + channel] & ~1) | int(combined_bits[i])
            bit_index += 3

        encoded_image = flat_image.reshape(image.shape).astype(np.uint8) #convert back to uint8
        _, buffer = cv2.imencode('.png', encoded_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        print("Encoding completed and image prepared in base64.")
        return base64_image
    except ValueError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def decode_message(encoded_image_path, password):
    """Decodes a message from an encoded image with random RGB selection."""
    try:
        encoded_image = cv2.imread(encoded_image_path)
        if encoded_image is None:
            raise ValueError("The encoded image could not be loaded.")

        seed = derive_seed(password)
        random.seed(seed)

        flat_image = encoded_image.flatten()
        length_bits = ""
        bit_index = 0
        for _ in range(16):
            channel = random.randint(0, 2)
            length_bits += str(flat_image[bit_index + channel] & 1)
            bit_index += 3

        message_length = int(length_bits, 2)
        message_bits = ""

        for _ in range(message_length * 8):
            channel = random.randint(0, 2)
            message_bits += str(flat_image[bit_index + channel] & 1)
            bit_index += 3

        message_bytes = bytes([int(message_bits[i:i + 8], 2) for i in range(0, len(message_bits), 8)])
        key = hashlib.sha256(password.encode()).digest()
        decoded_message = decrypt_message(message_bytes, key)
        print(f"Decoded message: {decoded_message}")
        return decoded_message

    except ValueError as e:
        print(f"Error: {e}")
        return f"Error: {e}"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"Error: {e}"


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        mode = request.form.get('mode')
        operation = request.form.get('operation')
        file = request.files.get('file')
        
        print(f"Received request - Mode: {mode}, Operation: {operation}, File: {file}")
        
        if not file:
            return jsonify({'success': False, 'message': 'No file provided.'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': 'Invalid file type.'}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            if mode == 'false':
                password = request.form.get('password')
                if not password:
                    return jsonify({'success': False, 'message': 'Password is required for image steganography.'}), 400
                
                if operation == 'encode':
                    secret_text = request.form.get('secret_text')
                    if not secret_text:
                        return jsonify({'success': False, 'message': 'Secret text is required for encoding.'}), 400
                    
                    base64_image = encode_message(filepath, secret_text, password, app.config['UPLOAD_FOLDER'])
                    return jsonify({
                        "success": True,
                        "encoded_image_base64": base64_image
                    })
                    
                elif operation == 'decode':
                    decoded_message = decode_message(filepath, password)
                    return jsonify({
                        "success": True,
                        "decoded_message": decoded_message
                    })
                    
            elif mode == 'true':
                if operation == 'encode':
                    secret_text = request.form.get('secret_text')
                    if not secret_text:
                        return jsonify({'success': False, 'message': 'Secret text is required for encoding.'}), 400
                    
                    encoded_audio = encode_phase(filepath, secret_text)
                    return jsonify({
                        "success": True,
                        "encoded_file_base64": encoded_audio
                    })
                    
                elif operation == 'decode':
                    decoded_message = decode_phase(filepath)
                    return jsonify({
                        "success": True,
                        "decoded_message": decoded_message
                    })
            else:
                return jsonify({'success': False, 'message': 'Invalid mode specified.'}), 400
                
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error processing the file: {str(e)}'}), 500
            
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)








