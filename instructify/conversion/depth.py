import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from transformers import pipeline
from asyncio import Semaphore

def get_relative_prominences(peaks, hist_array):
    """
    Calculate the relative prominence of each peak in a histogram array.

    Args:
        peaks (array-like): Indices of the peaks in the histogram array.
        hist_array (array-like): The histogram data array where peaks are identified.

    Returns:
        numpy.ndarray: An array of relative prominences for each peak. The relative prominence
        is calculated as the ratio of the peak's height to the maximum of the adjacent valley depths.
    """
    relative_prominences = []
    for i in range(len(peaks)):
        left_min = hist_array[peaks[i-1]:peaks[i]].min() if i > 0 else 0
        right_min = hist_array[peaks[i]:peaks[i+1]].min() if i < len(peaks) - 1 else 0
        relative_prominences.append(hist_array[peaks[i]] / max(left_min, right_min, 1e-6))
    return np.array(relative_prominences)

def create_image_from_array(group_array):
    """
    Create a grayscale image from a 2D numpy array by normalizing its values.

    Args:
        group_array (numpy.ndarray): A 2D array representing grouped data.

    Returns:
        PIL.Image.Image: A grayscale image created from the normalized group_array.
    """
    # Assuming 'group_array' is your numpy array of shape (1, 518, 420)
    group_array = group_array.squeeze()  # Remove the singleton dimension if it exists

    # Normalize the values to the range 0-255 for image representation
    normalized_array = (group_array - np.min(group_array)) / (np.max(group_array) - np.min(group_array))
    image_array = (normalized_array * 255).astype(np.uint8)  # Convert to uint8 for image creation
    img = Image.fromarray(image_array, 'L')  # 'L' mode is for grayscale
    return img

def find_and_group_peaks(two_d_array, prominence_factor=0.002, min_size_factor = 0.001, valley_threshold_ratio=1.6, max_peaks_to_consider = 15):
    """
    Identify peaks in the histogram of a 2D array and group the array elements based on these peaks.

    Args:
        two_d_array (numpy.ndarray): The 2D array from which to compute the histogram and find peaks.
        prominence_factor (float, optional): Factor to determine the required prominence of peaks.
            Calculated as hist_array.sum() * prominence_factor. Defaults to 0.002.
        min_size_factor (float, optional): Minimum peak height as a fraction of the total histogram sum
            to consider a peak valid. Defaults to 0.001.
        valley_threshold_ratio (float, optional): Threshold ratio for relative prominence to decide when
            to stop merging peaks. Defaults to 1.6.
        max_peaks_to_consider (int, optional): Maximum number of peaks to consider for grouping.
            Excess peaks are pruned based on prominence. Defaults to 15.

    Returns:
        tuple:
            numpy.ndarray: An array of the same shape as two_d_array where each element is assigned
            to a group based on the identified peaks.
            PIL.Image.Image: A grayscale image visualizing the group assignments.

        If insufficient peaks are found for grouping, returns (None, None).
    """
    # Get Peaks with find_peaks
    flat_array = np.array(two_d_array).flatten()
    hist_array, bin_edges = np.histogram(flat_array, bins=50)
    hist_median = np.median(hist_array)
    hist_array = hist_array / hist_median
    hist_array = np.insert(hist_array, 0, 0) # Add 0 so the first peak is not ignored
    
    peaks, info = find_peaks(hist_array, prominence=hist_array.sum()*prominence_factor)
    peaks = peaks - 1 # Subtract the added 0
    hist_array = hist_array[1:]
    # remove peaks that are below hist_array.sum()*min_size_factor
    peaks = peaks[hist_array[peaks] > hist_array.sum()*min_size_factor]
    if len(peaks) < 2: # We need at least 2 peaks to group the objects. 
        return None, None

    # take only the top max_peaks_to_consider
    if len(peaks) > max_peaks_to_consider:
        peak_prominences = info['prominences']
        peak_prominences, peaks = zip(*sorted(zip(peak_prominences, peaks), reverse=True)[:max_peaks_to_consider])
        peaks = np.sort(np.array(peaks))

    while len(peaks) > 2:
        # Calculate relative prominences
        relative_prominences = get_relative_prominences(peaks, hist_array)
        if relative_prominences.min() > valley_threshold_ratio:
            break
        # Remove the peak with the lowest relative prominence
        min_relative_prominence_index = relative_prominences.argmin()
        peaks = np.delete(peaks, min_relative_prominence_index)

    # Find valley points by looking for minima between peaks
    valley_indices = [(peaks[i] + peaks[i + 1]) // 2 for i in range(len(peaks) - 1)]
    valley_values = bin_edges[valley_indices]

    # Create group_array by assigning each value in the original array to a group based on nearest peak
    group_array = np.zeros_like(two_d_array, dtype=int)
    for i in range(len(valley_values) + 1):
        if i == 0:
            group_array[two_d_array <= valley_values[0]] = i
        elif i == len(valley_values):
            group_array[two_d_array > valley_values[-1]] = i
        else:
            group_array[(two_d_array > valley_values[i - 1]) & (two_d_array <= valley_values[i])] = i
    group_array = group_array.squeeze()
    return group_array, create_image_from_array(group_array)

########################################################## Depth Calculator ###########################################################

class DepthCalculator:
    def __init__(self):
        self.pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device="cuda:0")
        self.sem = Semaphore(1)
        
    async def predict(self, image, objects):
        """
        Estimate depth information for specified objects in an image.
        
        Args:
            image (PIL.Image or ndarray): The input image to process for depth estimation.
            objects (list of dict): A list of objects to evaluate depth. Each object dictionary should contain:
                - "mask" (ndarray): A binary mask of the object (same shape as `image`), where 1 represents the object's pixels.
                - "box" (tuple): The bounding box coordinates of the object, in the format (class_id, x1, y1, x2, y2).
                  Coordinates (x1, y1) and (x2, y2) should be normalized to [0, 1] relative to image dimensions.

        Returns:
            list of dict: A list of objects with depth information, where each object dictionary contains:
                - "depths" (ndarray): A sorted array of unique depth values for the object, in reverse order
                  (i.e., 0 represents the closest depth). If no depth information is available, `depths` will be an empty array.
        """
        depth_info = self.pipe(image)
        depth_image = depth_info["depth"]
        group_array, group_img = find_and_group_peaks(np.array(depth_info["depth"]))
        
        objects_with_group = []

        if group_array is None:
            for obj in objects:
                obj["depths"] = np.array([])
                objects_with_group.append(obj)
            return objects_with_group

        max_depth = group_array.max()
        for obj in objects:
            mask = obj["mask"]
            # if mask sum is 0, use the bounding box to create a mask
            if mask.sum() == 0:
                print("Warning: Mask is empty, using bounding box to create mask")
                _, x1, y1, x2, y2 = obj["box"]
                x1, y1, x2, y2 = int(x1*group_array.shape[1]), int(y1*group_array.shape[0]), int(x2*group_array.shape[1]), int(y2*group_array.shape[0])
                mask = np.zeros_like(group_array)
                mask[y1:y2, x1:x2] = 1

            # Use the mask to index directly into the depth array
            masked_depth = np.array(group_array)[mask > 0]  # Only take where mask is true

            # Clip the bottom and top 10%
            if masked_depth.size > 0:  # Check if there's any depth data within the mask
                lower_bound = np.percentile(masked_depth, 15)
                upper_bound = np.percentile(masked_depth, 85)
                masked_depth = np.clip(masked_depth, lower_bound, upper_bound)

                # Store unique depths in the object dictionary (Reverse depth so that 0 is the closest)
                obj["depths"] = np.sort((max_depth - np.unique(masked_depth)).tolist())
            else:
                # Handle case where mask might be empty
                obj["depths"] = np.array([])

            objects_with_group.append(obj)
        return objects_with_group