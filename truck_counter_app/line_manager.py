class LineManager:
    def __init__(self, pt1=None, pt2=None):
        """
        pt1, pt2: tuples of (x, y) relative coordinates (0.0 to 1.0)
        """
        self.pt1 = pt1
        self.pt2 = pt2

    def set_line(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2

    def clear_line(self):
        self.pt1 = None
        self.pt2 = None

    def get_absolute_points(self, width, height):
        if not self.pt1 or not self.pt2:
            return None, None
        p1 = (int(self.pt1[0] * width), int(self.pt1[1] * height))
        p2 = (int(self.pt2[0] * width), int(self.pt2[1] * height))
        return p1, p2

    def get_side(self, centroid, abs_pt1, abs_pt2):
        """
        Uses cross product to determine which side of the line the centroid is on.
        Returns > 0 for one side, < 0 for the other.
        """
        if not abs_pt1 or not abs_pt2:
            return 0
        
        x, y = centroid
        x1, y1 = abs_pt1
        x2, y2 = abs_pt2
        
        position = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        return position
