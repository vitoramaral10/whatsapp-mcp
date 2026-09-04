package main

import (
	"strings"
	"testing"

	waProto "go.mau.fi/whatsmeow/binary/proto"
	"google.golang.org/protobuf/proto"
)

const storedAudioURL = "https://mmg.whatsapp.net/v/t62.7117-24/791809332_4459727587672179_n.enc" +
	"?ccb=11-4&oh=01_Q5Aa5gGstfnHrhIfrrTdSoSI8SjByjMGs4&oe=6AC251A1&_nc_sid=5e03e0&mms3=true"

// whatsmeow builds the download URL as "https://<host><directPath>&hash=...", so
// the direct path has to arrive carrying its own query string. Dropping it left a
// URL with no "?" and no oh/oe signature, which the CDN answered with 403.
func TestExtractDirectPathFromURLKeepsTheSignature(t *testing.T) {
	got := extractDirectPathFromURL(storedAudioURL)

	if !strings.HasPrefix(got, "/v/t62.7117-24/") {
		t.Fatalf("expected a rooted media path, got %q", got)
	}
	if !strings.Contains(got, "?") {
		t.Fatalf("direct path lost its query string, so appending &hash= yields an unsigned URL: %q", got)
	}
	for _, param := range []string{"ccb=", "oh=", "oe=", "_nc_sid="} {
		if !strings.Contains(got, param) {
			t.Errorf("missing %q in direct path %q", param, got)
		}
	}
	if strings.Contains(got, "mmg.whatsapp.net") {
		t.Errorf("host should be stripped, whatsmeow supplies its own: %q", got)
	}
}

func TestExtractDirectPathFromURLLeavesUnparsableInput(t *testing.T) {
	if got := extractDirectPathFromURL("nonsense"); got != "nonsense" {
		t.Fatalf("expected the input back, got %q", got)
	}
}

func TestExtractMediaInfoCapturesDirectPath(t *testing.T) {
	tests := []struct {
		name         string
		msg          *waProto.Message
		wantType     string
		wantDirect   string
		wantFileName string
	}{
		{
			name: "audio",
			msg: &waProto.Message{AudioMessage: &waProto.AudioMessage{
				URL:        proto.String(storedAudioURL),
				DirectPath: proto.String("/v/t62.7117-24/791809332_n.enc?ccb=11-4&oh=abc&oe=def"),
				MediaKey:   []byte("key"),
				FileLength: proto.Uint64(7076),
			}},
			wantType:   "audio",
			wantDirect: "/v/t62.7117-24/791809332_n.enc?ccb=11-4&oh=abc&oe=def",
		},
		{
			name: "sticker used to be ignored entirely",
			msg: &waProto.Message{StickerMessage: &waProto.StickerMessage{
				URL:        proto.String("https://mmg.whatsapp.net/v/t62.1/sticker.enc?oh=z"),
				DirectPath: proto.String("/v/t62.1/sticker.enc?oh=z"),
			}},
			wantType:   "sticker",
			wantDirect: "/v/t62.1/sticker.enc?oh=z",
		},
		{
			name: "audio wrapped in a view-once container",
			msg: &waProto.Message{ViewOnceMessageV2: &waProto.FutureProofMessage{
				Message: &waProto.Message{AudioMessage: &waProto.AudioMessage{
					DirectPath: proto.String("/v/t62.7117-24/wrapped.enc?oh=y"),
				}},
			}},
			wantType:   "audio",
			wantDirect: "/v/t62.7117-24/wrapped.enc?oh=y",
		},
		{
			name: "document keeps its real name",
			msg: &waProto.Message{DocumentMessage: &waProto.DocumentMessage{
				FileName:   proto.String("resultado-exame.pdf"),
				DirectPath: proto.String("/v/t62.7119/doc.enc?oh=w"),
			}},
			wantType:     "document",
			wantDirect:   "/v/t62.7119/doc.enc?oh=w",
			wantFileName: "resultado-exame.pdf",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := extractMediaInfo(test.msg)

			if got.Type != test.wantType {
				t.Fatalf("type = %q, want %q", got.Type, test.wantType)
			}
			if got.DirectPath != test.wantDirect {
				t.Errorf("direct path = %q, want %q", got.DirectPath, test.wantDirect)
			}
			if test.wantFileName != "" && got.Filename != test.wantFileName {
				t.Errorf("filename = %q, want %q", got.Filename, test.wantFileName)
			}
		})
	}
}

func TestExtractMediaInfoIgnoresPlainText(t *testing.T) {
	if got := extractMediaInfo(&waProto.Message{Conversation: proto.String("oi")}); got.Type != "" {
		t.Fatalf("expected no media, got %q", got.Type)
	}
	if got := extractMediaInfo(nil); got.Type != "" {
		t.Fatalf("expected no media for a nil message, got %q", got.Type)
	}
}

// Filenames come from the clock at storage time, so messages stored in the same
// second collide; the message ID is what keeps two audios in one chat apart.
func TestSanitizeForFilename(t *testing.T) {
	tests := map[string]string{
		"3AFE0A5403DAC1B360AB": "3AFE0A5403DAC1B360AB",
		"AC46-76AD_09":         "AC46-76AD_09",
		"../../etc/passwd":     "______etc_passwd",
		"id:with@chars":        "id_with_chars",
	}
	for input, want := range tests {
		if got := sanitizeForFilename(input); got != want {
			t.Errorf("sanitizeForFilename(%q) = %q, want %q", input, got, want)
		}
	}
}
