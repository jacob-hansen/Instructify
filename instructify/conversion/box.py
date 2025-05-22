import torch
import numpy as np
import os
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils import merge_bboxes, masked_merge, custom_round
from utils import singular_to_plural, plural_to_singular
from collections import Counter, defaultdict
from asyncio import Semaphore
from conversion.depth import DepthCalculator  # Import the DepthCalculator

UNCOUNTABLE = set(["window", "drink", "tree", "building"])
MAX_COUNT_CLAIM = 8

def get_labels(box_string):
    return [plural_to_singular(label.strip()) for label in box_string.split('(')[0].split(',')]

def format_node(node, include_box_label=True, include_x1y1x2y2_label=False):
    box = node['box']
    label = box[0]
    x1, y1, x2, y2 = box[1:]
    x1y1x2y2_rounded = "[" + ", ".join([f"{custom_round(coord, precision=0.05):.2f}" for coord in box[1:]]) + "] "
    if not include_x1y1x2y2_label:
        x1y1x2y2_rounded = ""

    mask = node.get('mask', None)  # Get mask if available
    depths = node.get('depths', None)  # Get depths if available

    if not include_box_label:
        return label

    if mask is not None and mask.sum() > 0:
        # Calculate mask center and size
        y_indices, x_indices = np.where(mask)
        if len(x_indices) > 0 and len(y_indices) > 0:
            center_x = np.median(x_indices) / mask.shape[1]  # Normalize to [0,1]
            center_y = 1 - (np.median(y_indices) / mask.shape[0])  # Normalize to [0,1] and invert y
            pixel_size = (mask.sum() / (mask.shape[0] * mask.shape[1])) * 100  # Calculate percentage of pixels
        else:
            # Fallback to box center if mask is empty
            center_x = (x1 + x2) / 2
            center_y = 1 - (y1 + y2) / 2
            pixel_size = 0
    else:
        center_x = (x1 + x2) / 2
        center_y = 1 - (y1 + y2) / 2
        pixel_size = 0

    # Format the bounding box with descriptive labels
    formatted_box = (
        f"X: {custom_round(center_x, precision=0.05):.2f}, Y: {custom_round(center_y, precision=0.05):.2f}, "
        f"Pixel Size: {custom_round(pixel_size, precision=0.05):.1f}%"
    )

    # Include depth levels if available
    if depths is not None and len(depths) > 0:
        depth_levels = ', '.join(map(str, depths))
        formatted_box += f", Relative Depths: {depth_levels}"

    return f"{label} {x1y1x2y2_rounded}[{formatted_box}]"

def calculate_average_measurements(indent, nodes, num_display, plural, include_x1y1x2y2_label=False):
    """Calculate average measurements for a group of nodes"""
    centers_x = []
    centers_y = []
    widths = []
    heights = []
    sizes = []
    
    for node in nodes:
        box = node['box']
        x1, y1, x2, y2 = box[1:]
        width = x2 - x1
        height = y2 - y1
        
        # Get center coordinates
        if node.get('mask') is not None and node['mask'].sum() > 0:
            y_indices, x_indices = np.where(node['mask'])
            if len(x_indices) > 0 and len(y_indices) > 0:
                center_x = np.median(x_indices) / node['mask'].shape[1]
                center_y = 1 - (np.median(y_indices) / node['mask'].shape[0])
            else:
                center_x = (x1 + x2) / 2
                center_y = 1 - (y1 + y2) / 2
        else:
            center_x = (x1 + x2) / 2
            center_y = 1 - (y1 + y2) / 2
                
        centers_x.append(center_x)
        centers_y.append(center_y)
        
        # Calculate size if mask exists
        if node.get('mask') is not None:
            pixel_size = (node['mask'].sum() / (node['mask'].shape[0] * node['mask'].shape[1])) * 100
        else:
            pixel_size = 0
        sizes.append(pixel_size)
        
    # find x1y1x2y2_region_rounded by the min and max of the group
    region = [min([node['box'][1] for node in nodes]), min([node['box'][2] for node in nodes]),
              max([node['box'][3] for node in nodes]), max([node['box'][4] for node in nodes])]
    x1y1x2y2_region_rounded = "[" + ", ".join([f"{custom_round(coord, precision=0.05):.2f}" for coord in region]) + "] "
    if not include_x1y1x2y2_label:
        x1y1x2y2_region_rounded = ""
    
    # Use 'plural' and 'num_display' from context if needed
    return f"{indent}{num_display} ({plural}) {x1y1x2y2_region_rounded}[Average X: {custom_round(np.mean(centers_x), precision=0.05):.2f}, Average Y: {custom_round(np.mean(centers_y), precision=0.05):.2f}, Average Pixel Size: {np.mean(sizes):.1f}%]"

def count_all_types(node_list):
    type_counts = Counter()
    for node in node_list:
        labels = get_labels(node['box'][0])
        type_counts.update(labels)
        if node.get('children'):
            child_counts = count_all_types(node['children'])
            type_counts.update(child_counts)
    return type_counts

def depth_key_sort(node):
    if len(node.get('depths', [])) > 0:
        return np.mean(node['depths'])
    else:
        return 0

def size_key_sort(node):
    mask = node.get('mask')
    if mask is not None:
        return -mask.sum()
    return float('-inf')

def get_depth_range(node):
    depths = node.get('depths', [])
    if len(depths) > 0:
        return min(depths), max(depths)
    return None, None

def depths_overlap(range1, range2, grace=0):
    min1, max1 = range1
    min2, max2 = range2
    if None in (min1, max1, min2, max2):
        return False
    return max(min1 - grace, min2 - grace) <= min(max1 + grace, max2 + grace)

def position_close(node1, node2, threshold=2.0):
    _, x1, y1, x2, y2 = node1['box']
    _, x3, y3, x4, y4 = node2['box']
    
    # Check for overlap
    if x1 <= x4 and x2 >= x3 and y1 <= y4 and y2 >= y3:
        return True
        
    # If no overlap, check if distance is within threshold
    x_dist = min(abs(x1 - x4), abs(x2 - x3))
    y_dist = min(abs(y1 - y4), abs(y2 - y3))
    return x_dist <= threshold or y_dist <= threshold

class SortableItem:
    def __init__(self, text, sort_key):
        self.text = text
        self.sort_key = sort_key
    
    def __lt__(self, other):
        return self.sort_key < other.sort_key
    
class HierarchicalObjectOrganizer:
    def __init__(self, sam_model_name = "facebook/sam2-hiera-large", 
            initial_box_iou_threshold=0.95,
            merge_box_iou_threshold=0.6, 
            mask_containment_threshold=0.2):
        self.predictor = SAM2ImagePredictor.from_pretrained(sam_model_name)
        self.initial_box_iou_threshold = initial_box_iou_threshold
        self.merge_box_iou_threshold = merge_box_iou_threshold
        self.mask_containment_threshold = mask_containment_threshold
        self.semaphore = Semaphore(1)
        self.depth_calculator = DepthCalculator()  # Instantiate DepthCalculator
        self.max_resolution = 1920
        self.max_sam_boxes = 20
        self.semaphore = Semaphore(1)

    async def organize_objects_verbose(self, input_boxes, image_path, include_box_label=True, depth_calculation=False, include_x1y1x2y2_label=False):
        # Load and process the image
        img_pil = Image.open(image_path)
        img_pil.thumbnail((self.max_resolution, self.max_resolution))
        
        input_boxes = merge_bboxes(input_boxes, iou_threshold=self.initial_box_iou_threshold, format_the_labels=False)

        # Prepare input boxes for SAM
        input_boxes_torch = torch.tensor([box[1:] for box in input_boxes], dtype=torch.float32)
        input_boxes_torch[:, [0, 2]] *= img_pil.size[0]
        input_boxes_torch[:, [1, 3]] *= img_pil.size[1]

        # Set image once
        self.predictor.set_image(np.array(img_pil.convert("RGB")))
        
        # Process in batches (to avoid excessive memory usage with large number of boxes, the speed is nearly the same with batch size ~20)
        all_masks = []
        processed_boxes = []
        async with self.semaphore:
            for i in range(0, len(input_boxes), self.max_sam_boxes):
                batch_boxes = input_boxes_torch[i:i + self.max_sam_boxes]
                
                # Get masks from SAM for this batch
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    batch_masks, _, _ = self.predictor.predict(
                        box=batch_boxes.to(torch.int64),
                        multimask_output=False,
                    )
                    
                    # Handle single box case in the batch
                    if batch_boxes.shape[0] == 1:
                        batch_masks = [batch_masks]
                    
                    all_masks.extend(batch_masks)
                    processed_boxes.extend(input_boxes[i:i + self.max_sam_boxes])
                
                # Optional: Clear CUDA cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Merge the results using masked_merge
        merged_boxes, merged_masks = masked_merge(processed_boxes, all_masks, 
                                                iou_threshold=self.merge_box_iou_threshold, 
                                                format_the_labels=True)
        
        # Prepare objects with masks
        objects = []
        for i in range(len(merged_boxes)):
            obj = {"box": merged_boxes[i], "mask": merged_masks[i][0]}
            objects.append(obj)

        # Compute depths if depth_calculation is enabled
        if depth_calculation:
            objects = await self.depth_calculator.predict(img_pil, objects)

        # Organize objects hierarchically
        hierarchy = self._build_hierarchy(objects)

        # Format the hierarchy as a paragraph
        return self._format_hierarchy(hierarchy, include_box_label=include_box_label, include_x1y1x2y2_label=include_x1y1x2y2_label), merged_masks, merged_boxes

    async def organize_objects(self, input_boxes, image_path, include_box_label=True, depth_calculation=False):
        return (await self.organize_objects_verbose(input_boxes, image_path, include_box_label=include_box_label, depth_calculation=depth_calculation))[0]

    async def image_data_conversion(self, image_path, image_data, include_box_label=True, depth_calculation=False):
        aggregate_bbox = []
        for dataset in image_data:
            aggregate_bbox.extend(image_data[dataset]["bboxes"])

        if len(aggregate_bbox) > 0:
            return await self.organize_objects(aggregate_bbox, image_path, include_box_label=include_box_label, depth_calculation=depth_calculation)
        else:
            return ""

    def _build_hierarchy(self, objects):
        n = len(objects)
        containment_matrix = np.zeros((n, n), dtype=bool)

        # Build containment matrix
        for i in range(n):
            for j in range(n):
                object_i = objects[i]
                object_j = objects[j]
                label_i = object_i['box'][0].split(' (')[0]
                label_j = object_j['box'][0].split(' (')[0]
                if i == j or label_i == label_j:
                    continue

                # Check depth overlap (grace = 1)
                if not depths_overlap(get_depth_range(object_i), get_depth_range(object_j), grace=1):
                    continue

                box_i_area = (object_i['box'][3] - object_i['box'][1]) * (object_i['box'][4] - object_i['box'][2])
                box_j_area = (object_j['box'][3] - object_j['box'][1]) * (object_j['box'][4] - object_j['box'][2])
                contained_in_score = self._calculate_contained_in(object_i['mask'], object_j['mask'])
                if contained_in_score > self.mask_containment_threshold and box_i_area < box_j_area:
                    containment_matrix[i, j] = True
                    
        # Helper function to recursively build hierarchy
        used_indices = set()
        def build_node(index):
            if index in used_indices:
                return None
            used_indices.add(index)
            node = {
                "box": objects[index]['box'],
                "mask": objects[index]['mask'],  # Include mask in node
                "children": []
            }
            
            if 'depths' in objects[index]:
                node['depths'] = objects[index]['depths']

            for i in range(n):
                if containment_matrix[i, index]:
                    child_node = build_node(i)
                    if child_node:
                        node["children"].append(child_node)
            return node

        # Build hierarchy starting from root nodes
        hierarchy = []
        for i in range(n):
            if not containment_matrix[i].any() and i not in used_indices:  # This is a root node
                root_node = build_node(i)
                if root_node:
                    hierarchy.append(root_node)
        
        return hierarchy

    def _calculate_contained_in(self, mask1, mask2):
        """
        Calculate the 'contained_in' score, which is the fraction of mask1's area that is contained in mask2.
        """
        intersection = np.logical_and(mask1, mask2).sum()
        area_mask1 = mask1.sum()
        return intersection / area_mask1 if area_mask1 > 0 else 0

    def _format_hierarchy(self, hierarchy, level=0, include_box_label=True, include_x1y1x2y2_label=False):
        text = self._format_hierarchy_recuse(hierarchy, level=level, include_box_label=include_box_label, include_x1y1x2y2_label=include_x1y1x2y2_label)
        # add object counts to the end of the text
        all_types = count_all_types(hierarchy)
        type_counts = {singular_to_plural(type_): count for type_, count in all_types.items()}
        
        if len(type_counts) > 0:
            text += "\n\nObject Counts:"
            for type_, count in type_counts.items():
                text += f"\n\t{type_}: {count}"
        return text
    
    def _format_single_node(self, node, indent, include_box_label, include_x1y1x2y2_label):
        """Helper method to format a single node with its children."""
        if node.get('children'):
            return (f"{indent}{format_node(node, include_box_label, include_x1y1x2y2_label)}, with:\n" +
                    self._format_hierarchy_recuse(node['children'], 
                                            level=len(indent.split('->')[0])//4 + 1,
                                            include_box_label=include_box_label, include_x1y1x2y2_label=include_x1y1x2y2_label))
        return f"{indent}{format_node(node, include_box_label, include_x1y1x2y2_label)} X"

    def _determine_count_display(self, group, type_, plural):
        """Helper method to determine how to display the count of a group."""
        count = len(group)
        if count > MAX_COUNT_CLAIM:
            return "many"
        if any(plural in node['box'][0] for node in group):
            if count > MAX_COUNT_CLAIM:
                return "many"
            return "several"
        if any(uncountable in type_ for uncountable in UNCOUNTABLE):
            return "several"
        return str(count)
        
    def _format_hierarchy_recuse(self, hierarchy, level=0, include_box_label=True, include_x1y1x2y2_label=False):
        if not hierarchy:
            return ""
            
        # Sort hierarchy once, prioritizing size then depth
        hierarchy = sorted(hierarchy, key=lambda x: (
            size_key_sort(x),  # Already returns negative sum, so larger masks come first
            depth_key_sort(x)  # Smaller depths (closer) come first
        ))
        
        # Group nodes by type
        type_to_nodes = defaultdict(list)
        for node in hierarchy:
            labels = get_labels(node['box'][0])
            for label in labels:
                type_to_nodes[label].append(node)
        
        # Format output with sorted groups
        indent = '    ' * level + ('-> ' if level > 0 else '')
        output_items = []
        processed_nodes = set()

        for type_ in type_to_nodes:
            nodes = [node for node in type_to_nodes[type_] 
                    if id(node) not in processed_nodes]
            
            if not nodes:
                continue
                
            if len(nodes) > 1:
                # Group spatially close nodes
                spatial_groups = []
                for node in nodes:
                    node_range = get_depth_range(node)
                    group_found = False
                    
                    for group in spatial_groups:
                        # Check if node belongs in existing group
                        if all(depths_overlap(node_range, get_depth_range(g_node)) and 
                            position_close(node, g_node) for g_node in group):
                            group.append(node)
                            group_found = True
                            break
                            
                    if not group_found:
                        spatial_groups.append([node])

                # Process each spatial group
                for group in spatial_groups:
                    if len(group) > 1:
                        # Format multiple similar nodes together
                        plural = singular_to_plural(type_)
                        num_display = self._determine_count_display(group, type_, plural)
                        
                        if all(not node.get('children') for node in group):
                            # Format identical nodes with average measurements
                            text = calculate_average_measurements(indent, group, num_display, plural)
                            output_items.append(text)
                        else:
                            # Format group of different nodes
                            group_text = f"{indent}{num_display} ({plural}) {{\n"
                            for node in group:
                                node_text = self._format_single_node(node, indent + "    ", include_box_label, include_x1y1x2y2_label)
                                group_text += node_text + "\n"
                            group_text += f"{indent}}}"
                            output_items.append(group_text)
                            
                        for node in group:
                            processed_nodes.add(id(node))
                    else:
                        # Format single node in group
                        node = group[0]
                        if id(node) not in processed_nodes:
                            text = self._format_single_node(node, indent, include_box_label, include_x1y1x2y2_label)
                            output_items.append(text)
                            processed_nodes.add(id(node))
            else:
                # Format single node of this type
                node = nodes[0]
                if id(node) not in processed_nodes:
                    text = self._format_single_node(node, indent, include_box_label, include_x1y1x2y2_label)
                    output_items.append(text)
                    processed_nodes.add(id(node))
        return "\n\n".join(output_items)
