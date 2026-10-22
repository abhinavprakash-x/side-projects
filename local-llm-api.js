
async function askOllama(prompt)
{
    const response = await fetch("http://localhost:11434/api/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            model: "qwen3.5:9b",
            messages: [
                {
                    role: "user",
                    content: prompt
                }
            ],
            stream: true,
            think: false,
            keep_alive: "10m"
        })
    });

    if(!response.ok)
    {
        throw new Error(
            `Ollama returned HTTP ${response.status}: ${await response.text()}`
        );
    }

    if(!response.body)
    {
        throw new Error("Ollama returned an empty response stream.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let buffer = "";
    let answer = "";

    while(true)
    {
        const { value, done } = await reader.read();

        buffer += decoder.decode(value, { stream: !done });

        let newline;

        while((newline = buffer.indexOf("\n")) !== -1)
        {
            const line = buffer.slice(0, newline).trim();
            buffer = buffer.slice(newline + 1);

            if(!line)
            {
                continue;
            }

            const chunk = JSON.parse(line);
            const content = chunk.message?.content ?? "";

            process.stdout.write(content);
            answer += content;
        }

        if(done)
        {
            break;
        }
    }

    if(buffer.trim())
    {
        const chunk = JSON.parse(buffer);
        const content = chunk.message?.content ?? "";

        process.stdout.write(content);
        answer += content;
    }

    console.log();
    return answer;
}

async function main()
{
    try
    {
        console.log("Waiting for Qwen...\n");

        const answer = await askOllama(
            "Explain the difference between a process and a thread in three concise bullet points."
        );

        console.log("\n\nFinished generating.");
    }
    catch(error)
    {
        console.error("\nRequest failed:", error);
        process.exitCode = 1;
    }
}

main();