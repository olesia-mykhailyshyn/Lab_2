import numpy as np

#ф-я кодування
def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message]) #повідомлення перетворюється в вектор, де кожен символ перетворюється в його ASCII-код за допомогою ord(char)
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)
    return encrypted_vector


#ф-я розкодування
def decrypt_message(encrypted_vector, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    decrypted_vector = np.dot(np.linalg.inv(diagonalized_key_matrix), encrypted_vector)
    decrypted_message = ''.join([chr(int(round(value.real))) for value in decrypted_vector])
    return decrypted_message


# Генерація матриці ключа
key_matrix = np.random.randint(0, 256, (len("Hello, World!"), len("Hello, World!")))

original_message = "Hello, World!"
print("Original Message:", original_message)

encrypted_vector = encrypt_message(original_message, key_matrix)
print("Encrypted Message:", encrypted_vector)

decrypted_message = decrypt_message(encrypted_vector, key_matrix)
print("Decrypted Message:", decrypted_message)