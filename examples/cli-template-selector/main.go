package main

import (
	"context"
	"io"
	"os"
	"strings"

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/cli"
)

func main() {
	r := cli.NewRunner(
		cli.WithGraphComposer(func(ctx context.Context, cfg cli.Config, p cli.ProviderKind, base textual.Processor[textual.String], stderr io.Writer) (textual.Processor[textual.String], error) {
			local := textual.ProcessorFunc[textual.String](func(ctx context.Context, in <-chan textual.String) <-chan textual.String {
				out := make(chan textual.String)
				go func() {
					defer close(out)
					for {
						select {
						case <-ctx.Done():
							return
						case item, ok := <-in:
							if !ok {
								return
							}
							// Emit a single final value (a "snapshot") so the default
							// streamer prints it once.
							out <- textual.String{}.FromUTF8String("pong").WithIndex(item.GetIndex())
						}
					}
				}()
				return out
			})

			router := textual.NewRouter[textual.String](textual.RoutingStrategyFirstMatch)
			router.AddRoute(func(ctx context.Context, item textual.String) bool {
				return strings.TrimSpace(item.UTF8String()) == "/ping"
			}, local)

			// Catch-all route: the real provider (OpenAI/Ollama).
			router.AddProcessor(base)

			return router, nil
		}),
	)

	os.Exit(r.Run(os.Args, os.Stdout, os.Stderr))
}
