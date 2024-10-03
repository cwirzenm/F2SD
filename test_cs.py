from object_detection_yolov8x import YoloV8X
import numpy as np
import os


def cosine_similarity(vec1, vec2):
    # Ensure that the input vectors are numpy arrays
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # Calculate the dot product of the two vectors
    dot_product = np.dot(vec1, vec2)

    # Calculate the norm (magnitude) of each vector
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # Check if any of the norms are zero to avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        raise ValueError("One of the vectors is zero-vector, cannot calculate cosine similarity")

    # Calculate the cosine similarity
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)

    return cosine_sim


if __name__ == '__main__':
    model = YoloV8X()

    # Geralt of Rivia
    path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test2/set_1/'
    embeddings = model(path, show=False, return_cropped_frames=False)
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    print(f"Geralt Cosine Similarity: {similarity}")

    path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test2/set_2/'
    embeddings = model(path, show=False, return_cropped_frames=False)
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    print(f"Fake Geralt Cosine Similarity: {similarity}")
