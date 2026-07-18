import pandas as pd
from pathlib import Path
from PIL import Image
from config import YFCC4K_IMG_DIR, YFCC4K_GPS_CSV


class YFCC4KDataset:
    """
    YFCC4K evaluation dataset.
    CSV columns: photo_id, lat, lon, img_url. For dataset smoke runs, also
    accepts common aliases such as IMG_ID/filename/image and LAT/LON.
    Images are expected at: {img_dir}/{photo_id}.jpg, or at the exact filename
    when the CSV image id already includes an extension.
    """

    def __init__(self, img_dir: str = YFCC4K_IMG_DIR, gps_csv: str = YFCC4K_GPS_CSV):
        self.img_dir = Path(img_dir)
        self.meta = self._normalize_meta(pd.read_csv(gps_csv))
        self.meta = self.meta[
            self.meta["photo_id"].apply(lambda pid: self._image_path(pid).exists())
        ].reset_index(drop=True)
        print(f"[Dataset] {len(self.meta)} images found in {img_dir}")

    @staticmethod
    def _pick_column(columns, candidates):
        lowered = {str(c).strip().lower(): c for c in columns}
        for name in candidates:
            if name in lowered:
                return lowered[name]
        return None

    @classmethod
    def _normalize_meta(cls, meta):
        meta = meta.rename(columns={c: str(c).strip() for c in meta.columns})
        photo_col = cls._pick_column(
            meta.columns,
            ["photo_id", "img_id", "image_id", "filename", "file", "image", "img", "path", "name", "id"],
        )
        lat_col = cls._pick_column(meta.columns, ["lat", "latitude", "gps_lat"])
        lon_col = cls._pick_column(meta.columns, ["lon", "lng", "longitude", "gps_lon"])
        missing = []
        if photo_col is None:
            missing.append("photo_id/img_id/filename")
        if lat_col is None:
            missing.append("lat/latitude")
        if lon_col is None:
            missing.append("lon/lng/longitude")
        if missing:
            raise ValueError(
                "GPS CSV missing required columns: "
                + ", ".join(missing)
                + f". Available columns: {list(meta.columns)}"
            )
        return meta.rename(columns={photo_col: "photo_id", lat_col: "lat", lon_col: "lon"})

    def _image_path(self, photo_id):
        name = str(photo_id)
        path = Path(name)
        if path.suffix:
            return self.img_dir / path.name
        return self.img_dir / f"{name}.jpg"

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_path = self._image_path(row["photo_id"])
        image = Image.open(img_path).convert("RGB")
        return {
            "photo_id": str(row["photo_id"]),
            "image": image,
            "gt_lat": float(row["lat"]),
            "gt_lon": float(row["lon"]),
            "img_path": str(img_path),
        }
