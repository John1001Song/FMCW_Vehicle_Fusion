class RadarDataset(torch.utils.data.Dataset):
    def __init__(self, npy_file):
        self.data = np.load(npy_file, allow_pickle=True)
        self.points = self.data['points']  # 3D coordinates
        self.velocities = self.data['Velocity']
        self.range = self.data['Range']
        self.bearings = self.data['Bearing']
        self.intensity = self.data['Intensity']
    
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, idx):
        return {
            'points': torch.tensor(self.points[idx], dtype=torch.float32),
            'velocity': torch.tensor(self.velocities[idx], dtype=torch.float32),
            'range': torch.tensor(self.range[idx], dtype=torch.float32),
            'bearing': torch.tensor(self.bearings[idx], dtype=torch.float32),
            'intensity': torch.tensor(self.intensity[idx], dtype=torch.float32),
        }
