import numpy as np

class RigidAlignment:
    """
    A class for rigid alignment of landmarks using the Kabsch-Umeyama algorithm.
    """

    @staticmethod
    def kabsch_umeyama(A: np.ndarray, B: np.ndarray) -> tuple:
        """
        Computes the optimal rigid transformation (rotation, scale, and translation) 
        to align two sets of points using the Kabsch-Umeyama algorithm.

        Args:
            A (np.ndarray): Reference points, shape (N, D), where N is the number of points, D is the dimensionality.
            B (np.ndarray): Points to be aligned, shape (N, D).

        Returns:
            tuple: Rotation matrix R, scaling factor c, and translation vector t.
        """
        if A.shape != B.shape:
            raise ValueError("Input arrays A and B must have the same shape.")
        
        n, m = A.shape

        # Compute centroids
        EA = np.mean(A, axis=0)
        EB = np.mean(B, axis=0)

        # Compute variance of A
        VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

        # Compute cross-covariance matrix
        H = ((A - EA).T @ (B - EB)) / n
        U, D, VT = np.linalg.svd(H)

        # Compute determinant to ensure right-handed coordinate system
        d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
        S = np.diag([1] * (m - 1) + [d])

        # Compute rotation matrix, scaling factor, and translation vector
        R = U @ S @ VT
        c = VarA / np.trace(np.diag(D) @ S)
        t = EA - c * R @ EB

        return R, c, t

    @staticmethod
    def get_transformed_vector(reference_landmarks: np.ndarray, single_landmarks: np.ndarray) -> np.ndarray:
        """
        Transforms a set of landmarks to align with the reference landmarks.

        Args:
            reference_landmarks (np.ndarray): Reference landmarks, shape (N, 2) or (N, 3).
            single_landmarks (np.ndarray): Landmarks to be transformed, shape (N, 2) or (N, 3).

        Returns:
            np.ndarray: Transformed landmarks, same shape as input landmarks.
        """
        if reference_landmarks.shape != single_landmarks.shape:
            raise ValueError("Reference and single landmarks must have the same shape.")
        
        # Extract center coordinates for alignment
        ref_centers = RigidAlignment.get_center_coordinates(reference_landmarks)
        single_centers = RigidAlignment.get_center_coordinates(single_landmarks)

        # Apply the Kabsch-Umeyama algorithm
        R, c, t = RigidAlignment.kabsch_umeyama(ref_centers, single_centers)

        # Transform all landmarks
        transformed_landmarks = np.array([t + c * R @ point for point in single_landmarks])
        return transformed_landmarks

    @staticmethod
    def get_center_coordinates(landmarks: np.ndarray) -> np.ndarray:
        """
        Extracts the coordinates of specific landmarks for alignment.

        Args:
            landmarks (np.ndarray): All landmarks, shape (N, 2) or (N, 3).

        Returns:
            np.ndarray: Selected center landmarks, shape (4, 2) or (4, 3).
        """
        # Indices for center landmarks
        center_indices = [1, 2, 7, 8]
        if max(center_indices) >= landmarks.shape[0]:
            raise ValueError(f"Landmarks array must have at least {max(center_indices) + 1} points.")

        center_landmarks = landmarks[center_indices]
        return np.asarray(center_landmarks)
