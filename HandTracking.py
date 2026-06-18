import os
import math
import urllib.request
import time
import cv2
import mediapipe as mp


class OvladaniGesty:
    URL_MODELU = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    CESTA_K_MODELU = "hand_landmarker.task"
    KOSTRA_RUKY = [
        (0,1), (1,2), (2,3), (3,4),
        (0,5), (5,6), (6,7), (7,8),
        (5,9), (9,10), (10,11), (11,12),
        (9,13), (13,14), (14,15), (15,16),
        (13,17), (17,18), (18,19), (19,20),
        (0,17)
    ]

    def __init__(self):

        if not os.path.exists(self.CESTA_K_MODELU):
            print("Model nebyl nalezen. Stahuji...")
            urllib.request.urlretrieve(self.URL_MODELU, self.CESTA_K_MODELU)

        nastaveni = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=self.CESTA_K_MODELU
            ),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            result_callback=self._uloz_data_ruky
        )

        self.detektor = mp.tasks.vision.HandLandmarker.create_from_options(
            nastaveni
        )

        self.data_ruky = None

        self.zoom = 1.0
        self.cilovy_zoom = 1.0

        self.cislo_snimku = 0
        self.pomer_prstu = 1.3

        self.ukonceni_aktivni = False
        self.blokace_do = 0.0
        self.casovac_start = 0.0

    def _uloz_data_ruky(self, vysledek, vystupni_obrazek, cas_ms):
        self.data_ruky = vysledek

    def _zpracuj_gesta_a_vykresli(self, snimek, sirka, vyska):

        if not self.data_ruky or not self.data_ruky.hand_landmarks:
            return False

        aktualni_cas = time.time()
        ruce = self.data_ruky.hand_landmarks

        for ruka in ruce:

            for spojeni in self.KOSTRA_RUKY:
                bod1 = (
                    int(ruka[spojeni[0]].x * sirka),
                    int(ruka[spojeni[0]].y * vyska)
                )

                bod2 = (
                    int(ruka[spojeni[1]].x * sirka),
                    int(ruka[spojeni[1]].y * vyska)
                )

                cv2.line(snimek, bod1, bod2, (255, 0, 0), 1)

            for bod in ruka:
                cv2.circle(
                    snimek,
                    (int(bod.x * sirka), int(bod.y * vyska)),
                    4,
                    (0, 230, 255),
                    -1
                )

        if self.ukonceni_aktivni:

            if aktualni_cas - self.casovac_start >= 5.0:
                self.ukonceni_aktivni = False
                self.blokace_do = aktualni_cas + 3.0
                return False

            hlavni_ruka = ruce[0]

            if (
                hlavni_ruka[4].y < hlavni_ruka[3].y and
                hlavni_ruka[8].y > hlavni_ruka[6].y and
                hlavni_ruka[12].y > hlavni_ruka[10].y and
                hlavni_ruka[16].y > hlavni_ruka[14].y
            ):
                return True

            return False

        if len(ruce) == 2 and aktualni_cas > self.blokace_do:

            pocet_otevrenych = 0

            for ruka in ruce:
                if (
                    ruka[8].y < ruka[6].y and
                    ruka[12].y < ruka[10].y and
                    ruka[16].y < ruka[14].y and
                    ruka[20].y < ruka[18].y
                ):
                    pocet_otevrenych += 1

            if pocet_otevrenych == 2:
                self.ukonceni_aktivni = True
                self.casovac_start = aktualni_cas
                return False

        
        hlavni_ruka = ruce[0]

        vzdalenost_prstu = math.hypot(
            (hlavni_ruka[4].x - hlavni_ruka[8].x) * sirka,
            (hlavni_ruka[4].y - hlavni_ruka[8].y) * vyska
        )

        sirka_dlane = math.hypot(
            (hlavni_ruka[5].x - hlavni_ruka[17].x) * sirka,
            (hlavni_ruka[5].y - hlavni_ruka[17].y) * vyska
        )

        if sirka_dlane > 0:
            self.pomer_prstu = max(
                0.3,
                min(vzdalenost_prstu / sirka_dlane, 1.3)
            )

            self.cilovy_zoom = (
                1.0 +
                (1.0 - (self.pomer_prstu - 0.3) / 1.0) * 3.0
            )

        palec = (
            int(hlavni_ruka[4].x * sirka),
            int(hlavni_ruka[4].y * vyska)
        )

        ukazovacek = (
            int(hlavni_ruka[8].x * sirka),
            int(hlavni_ruka[8].y * vyska)
        )

        barva = int(((self.pomer_prstu - 0.3) / 1.0) * 255)

        cv2.line(
            snimek,
            palec,
            ukazovacek,
            (0, barva, 255 - barva),
            3
        )

        return False

    def _vykresli_okno_ukonceni(self, snimek, sirka, vyska):

        prekryti = snimek.copy()

        cv2.rectangle(
            prekryti,
            (0, 0),
            (sirka, vyska),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(prekryti, 0.6, snimek, 0.4, 0, snimek)

        font = cv2.FONT_HERSHEY_SIMPLEX

        nadpis = "Chceš ukončit aplikaci?"
        podnadpis = "Pro potvrzení ukaž palec nahoru"

        (sirka_textu, _), _ = cv2.getTextSize(
            nadpis,
            font,
            0.8,
            2
        )

        cv2.putText(
            snimek,
            nadpis,
            ((sirka - sirka_textu) // 2, (vyska // 2) - 20),
            font,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            snimek,
            podnadpis,
            ((sirka - 320) // 2, (vyska // 2) + 20),
            font,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    def spust(self):

        kamera = cv2.VideoCapture(0)

        while kamera.isOpened():

            uspesne, snimek = kamera.read()

            if not uspesne:
                break

            snimek = cv2.flip(snimek, 1)

            vyska, sirka, _ = snimek.shape

            obrazek = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(snimek, cv2.COLOR_BGR2RGB)
            )

            self.cislo_snimku += 1

            self.detektor.detect_async(
                obrazek,
                self.cislo_snimku
            )

            ukoncit = self._zpracuj_gesta_a_vykresli(
                snimek,
                sirka,
                vyska
            )

            if ukoncit:
                break

            if not self.ukonceni_aktivni:

                self.zoom += (
                    self.cilovy_zoom - self.zoom
                ) * 0.2

                if self.zoom > 1.01:

                    nova_sirka = int(sirka / self.zoom)
                    nova_vyska = int(vyska / self.zoom)

                    posun_x = (sirka - nova_sirka) // 2
                    posun_y = (vyska - nova_vyska) // 2

                    snimek = cv2.resize(
                        snimek[
                            posun_y:posun_y + nova_vyska,
                            posun_x:posun_x + nova_sirka
                        ],
                        (sirka, vyska)
                    )

                aktualni_cas = time.time()

                if aktualni_cas < self.blokace_do:
                    cv2.putText(
                        snimek,
                        f"Ukonceni blokovano ({int(self.blokace_do - aktualni_cas) + 1}s)",
                        (sirka - 260, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA
                    )

                cv2.putText(
                    snimek,
                    f"Priblizeni: {self.zoom:.2f}x",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            else:
                self._vykresli_okno_ukonceni(
                    snimek,
                    sirka,
                    vyska
                )

            cv2.imshow(
                "Ovladani zoomu gesty",
                snimek
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        kamera.release()
        cv2.destroyAllWindows()
        self.detektor.close()


if __name__ == "__main__":
    aplikace = OvladaniGesty()
    aplikace.spust()
