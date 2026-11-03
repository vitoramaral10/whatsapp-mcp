import base64
import os
import unittest

os.environ["WHATSAPP_MEDIA_HOST_PATH"] = "/host/store"

import main  # noqa: E402


class MediaPathTest(unittest.TestCase):
    def test_container_path_undoes_host_path(self):
        container = "/app/whatsapp-bridge/store/chat/img.jpg"
        self.assertEqual(main._container_path(main._host_visible_path(container)), container)

    def test_container_path_leaves_other_paths(self):
        self.assertEqual(main._container_path("/host/storeX/a.jpg"), "/host/storeX/a.jpg")
        self.assertEqual(main._container_path("/app/envios/a.jpg"), "/app/envios/a.jpg")


class MediaFileTest(unittest.TestCase):
    def test_base64_written_under_its_filename_and_removed(self):
        content = base64.b64encode(b"%PDF-1.4").decode()
        with main._media_file("", content, "../relatorio.pdf") as path:
            self.assertEqual(os.path.basename(path), "relatorio.pdf")
            with open(path, "rb") as f:
                self.assertEqual(f.read(), b"%PDF-1.4")
        self.assertFalse(os.path.exists(path))

    def test_rejects_missing_extension_and_bad_base64(self):
        for content, filename in (("aGk=", "semextensao"), ("aGk=", ""), ("não é base64", "a.txt")):
            with self.assertRaises(ValueError):
                with main._media_file("", content, filename):
                    pass


if __name__ == "__main__":
    unittest.main()
