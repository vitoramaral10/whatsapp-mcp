package main

import (
	"strings"
	"testing"

	waProto "go.mau.fi/whatsmeow/binary/proto"
	"google.golang.org/protobuf/proto"
)

// Every case here used to return "" from extractTextContent, which makes
// handleMessage drop the message before it reaches the database — the reason a
// bot menu looked like the bot had answered nothing.
func TestExtractTextContentRendersInteractiveMenus(t *testing.T) {
	tests := []struct {
		name  string
		msg   *waProto.Message
		want  []string
		empty bool
	}{
		{
			name: "plain conversation still works",
			msg:  &waProto.Message{Conversation: proto.String("Boa tarde")},
			want: []string{"Boa tarde"},
		},
		{
			name: "list message renders rows with their ids",
			msg: &waProto.Message{ListMessage: &waProto.ListMessage{
				Title:      proto.String("Centro Médico"),
				ButtonText: proto.String("Ver opções"),
				Sections: []*waProto.ListMessage_Section{{
					Title: proto.String("Agendamento"),
					Rows: []*waProto.ListMessage_Row{
						{Title: proto.String("Agendar Consultas"), RowID: proto.String("opt_1")},
						{Title: proto.String("Agendar Exames"), Description: proto.String("laboratório"), RowID: proto.String("opt_2")},
					},
				}},
			}},
			want: []string{"Centro Médico", "Agendamento", "- Agendar Consultas [opt_1]", "- Agendar Exames: laboratório [opt_2]", "Ver opções"},
		},
		{
			name: "buttons message renders each button",
			msg: &waProto.Message{ButtonsMessage: &waProto.ButtonsMessage{
				ContentText: proto.String("Os dados estão corretos?"),
				Buttons: []*waProto.ButtonsMessage_Button{
					{ButtonID: proto.String("yes"), ButtonText: &waProto.ButtonsMessage_Button_ButtonText{DisplayText: proto.String("SIM")}},
					{ButtonID: proto.String("no"), ButtonText: &waProto.ButtonsMessage_Button_ButtonText{DisplayText: proto.String("NÃO")}},
				},
			}},
			want: []string{"Os dados estão corretos?", "- SIM [yes]", "- NÃO [no]"},
		},
		{
			name: "native flow single_select is parsed out of the json payload",
			msg: &waProto.Message{InteractiveMessage: &waProto.InteractiveMessage{
				Body: &waProto.InteractiveMessage_Body{Text: proto.String("Escolha uma opção")},
				InteractiveMessage: &waProto.InteractiveMessage_NativeFlowMessage_{
					NativeFlowMessage: &waProto.InteractiveMessage_NativeFlowMessage{
						Buttons: []*waProto.InteractiveMessage_NativeFlowMessage_NativeFlowButton{{
							Name: proto.String("single_select"),
							ButtonParamsJSON: proto.String(
								`{"title":"Menu","sections":[{"title":"Atendimento","rows":[{"title":"Ouvidoria","id":"3"}]}]}`),
						}},
					},
				},
			}},
			want: []string{"Escolha uma opção", "Menu", "Atendimento", "- Ouvidoria [3]"},
		},
		{
			name: "template quick replies render as options",
			msg: &waProto.Message{TemplateMessage: &waProto.TemplateMessage{
				Format: &waProto.TemplateMessage_HydratedFourRowTemplate_{
					HydratedFourRowTemplate: &waProto.TemplateMessage_HydratedFourRowTemplate{
						HydratedContentText: proto.String("Confirma o cancelamento?"),
						HydratedButtons: []*waProto.HydratedTemplateButton{{
							HydratedButton: &waProto.HydratedTemplateButton_QuickReplyButton{
								QuickReplyButton: &waProto.HydratedTemplateButton_HydratedQuickReplyButton{
									DisplayText: proto.String("Confirmar"), ID: proto.String("ok"),
								},
							},
						}},
					},
				},
			}},
			want: []string{"Confirma o cancelamento?", "- Confirmar [ok]"},
		},
		{
			name: "ephemeral wrapper is unwrapped",
			msg: &waProto.Message{EphemeralMessage: &waProto.FutureProofMessage{
				Message: &waProto.Message{Conversation: proto.String("mensagem temporária")},
			}},
			want: []string{"mensagem temporária"},
		},
		{
			name: "image caption is no longer lost",
			msg: &waProto.Message{ImageMessage: &waProto.ImageMessage{
				Caption: proto.String("resultado do exame"),
			}},
			want: []string{"resultado do exame"},
		},
		{
			name: "list reply records what we selected",
			msg: &waProto.Message{ListResponseMessage: &waProto.ListResponseMessage{
				Title:             proto.String("Agendar Consultas"),
				SingleSelectReply: &waProto.ListResponseMessage_SingleSelectReply{SelectedRowID: proto.String("opt_1")},
			}},
			want: []string{"Agendar Consultas", "opt_1"},
		},
		{
			name:  "a message with nothing readable stays empty",
			msg:   &waProto.Message{},
			empty: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := extractTextContent(test.msg)

			if test.empty {
				if got != "" {
					t.Fatalf("expected no text, got %q", got)
				}
				return
			}
			if got == "" {
				t.Fatal("message was dropped: extractTextContent returned an empty string")
			}
			for _, want := range test.want {
				if !strings.Contains(got, want) {
					t.Errorf("missing %q in:\n%s", want, got)
				}
			}
		})
	}
}

// The depth cap has to hold even if a message nests containers into each other.
func TestExtractTextContentStopsRecursing(t *testing.T) {
	msg := &waProto.Message{Conversation: proto.String("fundo")}
	for i := 0; i < 10; i++ {
		msg = &waProto.Message{EphemeralMessage: &waProto.FutureProofMessage{Message: msg}}
	}
	if got := extractTextContent(msg); got != "" {
		t.Fatalf("expected the depth cap to give up, got %q", got)
	}
}

func TestUnhandledMessageFieldsNamesTheType(t *testing.T) {
	msg := &waProto.Message{ReactionMessage: &waProto.ReactionMessage{Text: proto.String("👍")}}
	if got := unhandledMessageFields(msg); !strings.Contains(got, "reactionMessage") {
		t.Fatalf("expected the field name in %q", got)
	}
	if got := unhandledMessageFields(nil); got != "<nil>" {
		t.Fatalf("expected <nil>, got %q", got)
	}
}
